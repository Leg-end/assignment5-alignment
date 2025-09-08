import os
os.environ["HF_ENDPOINT"] = 'https:/hf-mirror.com'
os.environ["HF_HOME"] = "/data/lanyun/worksapce/assignment5-alignment/models"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import comet_ml as comet
from typing import Literal, Callable
from vllm import SamplingParams, LLM, RequestOutput
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.data_utils import get_data_loader, gsm8k_reward_fn, SFTDataset
from cs336_alignment.train_sft import init_vllm, load_policy_into_vllm_instance, evaluate
from cs336_alignment.utils import set_logger, pad
from torch.utils.data import DataLoader
from tests.adapters import run_compute_group_normalized_rewards, run_grpo_microbatch_train_step,\
    run_get_response_log_probs, run_tokenize_prompt_and_output, run_compute_policy_gradient_loss,\
    run_masked_mean
from trl.trainer.grpo_trainer import RepeatSampler, shuffle_tensor_dict, split_tensor_dict
from trl.trainer.utils import selective_log_softmax

import torch
import logging

@torch.inference_mode()
def get_old_logprobs(model: PreTrainedModel,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     logits_to_keep: int | torch.Tensor,
                     batch_size: int | None = None,
                     temperature: float = 1.0):
    """
    copied from trl.trainer.GRPOTrainer
    """
    batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
    all_logps = []
    for i in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[i : i + batch_size]
        attention_mask_batch = attention_mask[i : i + batch_size]

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids_batch = input_ids_batch[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / temperature
        logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
        all_logps.append(logps)
    return torch.cat(all_logps, dim=0)


def sample_rollout(batch: dict[str, list[str]],
                   vllm_old_policy: LLM,
                   tokenizer: PreTrainedTokenizerBase,
                   reward_fn: Callable,
                   hf_old_policy: PreTrainedModel,
                   micro_batch_size: int,
                   advantage_eps: float = 1e-6,
                   group_size: int = 8,
                   use_std_normalization: bool = True,
                   off_policy: bool = False,
                   use_vllm_logprob: bool = True) -> tuple[dict[str, torch.Tensor | list[str]], dict[str, float]]:
    """
    off-policy happens when epochs_per_rollout_batch > 1 or n_microbatches_per_rollout_batch > gradient_accumulation_steps
    i.e. when rollout_batch_size > train_batch_size, as policy will be updated after consuming train_batch_size samples
    then new policy will continue training on the remaining (rollout_batch_size - train_batch_size) samples, which are
    generated from old policy.
    batch: has its value with length n_microbatches_per_rollout_batch * micro_train_batch_size
    """
    if use_vllm_logprob:
        device = vllm_old_policy.llm_engine.model_executor.driver_worker.model_runner.model.device
    else:
        device = hf_old_policy.device
    sampling_params = SamplingParams(n=group_size,
                                     temperature=1.0,
                                     top_p=1.0,
                                     min_tokens=4, # As in Expiter, disallow empty string responses
                                     max_tokens=1024,
                                     stop=['</answer>'],
                                     include_stop_str_in_output=True,
                                     logprobs=1 if use_vllm_logprob else None)
    # Sample a n_prompts_per_rollout_batch of questions D_b from D
    prompt_strs, answer_strs = batch['prompt'], batch['answer']
    # Sample group_size response for each question in the batch, rollout_batch_size = n_prompts_per_rollout_batch * group_size
    request_outputs: list[RequestOutput] = vllm_old_policy.generate(prompt_strs, sampling_params)
    rollout_responses = []
    rollout_prompt_ids = []
    rollout_response_ids = []
    for request_output in request_outputs:
        output = request_output.outputs[0]
        rollout_responses.append(output.text.strip())
        rollout_prompt_ids.append(torch.tensor(request_output.prompt_token_ids,
                                               device=device, dtype=torch.long))
        rollout_response_ids.append(torch.tensor(output.token_ids,
                                                 device=device, dtype=torch.long))
    rollout_prompt_ids, rollout_prompt_masks = pad(rollout_prompt_ids,
                                                   padding_value=tokenizer.pad_token_id,
                                                   padding_side='left')
    rollout_response_ids, rollout_response_masks = pad(rollout_response_ids,
                                                       padding_value=tokenizer.pad_token_id,
                                                       padding_side='right')
    response_mask = torch.cat([torch.zeros_like(rollout_prompt_masks), rollout_response_masks], dim=1)
    input_ids = torch.cat([rollout_prompt_ids, rollout_response_ids], dim=1)
    if off_policy:
        if use_vllm_logprob: # Issue, logprobs from vllm inconsistent with that from transformers
            old_log_probs = []
            for request_output in request_outputs:
                output = request_output.outputs[0]
                log_probs = [logprob[token_id].logprob for token_id, logprob in zip(output.token_ids, output.logprobs)]
                old_log_probs.append(torch.tensor(log_probs, dtype=torch.float32, device=device))
            max_len = max(len(log_probs) for log_probs in old_log_probs)
            # right padding the old_log_probs
            pad_old_log_probs = torch.full((len(old_log_probs), max_len), -float('inf'),
                                           dtype=torch.float32, device=device)
            for i, log_probs in enumerate(old_log_probs):
                seq_len = len(log_probs)
                pad_old_log_probs[i, :seq_len] = log_probs
            old_log_probs = pad_old_log_probs
        else:
            attention_mask = torch.cat([rollout_prompt_masks, rollout_response_masks], dim=1)
            logits_to_keep = rollout_response_ids.size(1)  # we only need to compute the logits for the completion tokens
            old_log_probs = get_old_logprobs(hf_old_policy, input_ids, attention_mask, logits_to_keep, batch_size=micro_batch_size)
    else:
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
        # per_token_logps.detach() instead.
        old_log_probs = None
    # Compute rewards for each sampled response by running reward function
    # and advantages with group normalization
    advantages, raw_rewards, metadata = run_compute_group_normalized_rewards(
        reward_fn=reward_fn, rollout_responses=rollout_responses,
        repeated_ground_truths=answer_strs,
        group_size=group_size, advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization
    )
    # TODO filter out wrong outputs with low rewards
    rollouts = {
        "input_ids": input_ids,
        "labels": rollout_response_ids,
        "response_masks": response_mask,
        "advantages": advantages,
        "raw_rewards": raw_rewards,
        "old_log_probs": old_log_probs
    }
    return rollouts, metadata


def evaluate(eval_loader: DataLoader,
             tokenizer: PreTrainedTokenizerBase,
             old_policy: LLM,
             new_policy: PreTrainedModel,
             reward_fn: Callable,
             advantage_eps: float = 1e-6,
             eval_batch_size: int = 16,
             loss_type: Literal[
                "no_baseline",
                "reinforce_with_baseline",
                "grpo_clip",
             ] = "reinforce_with_baseline",
             cliprange: float = 0.2,
             use_std_normalization: bool = True):
    load_policy_into_vllm_instance(new_policy, old_policy)
    total_loss = 0.0
    metadata = {}
    for batch in eval_loader:
        rollout, data_metadata = sample_rollout(batch=batch,
                                                vllm_old_policy=old_policy,
                                                hf_old_policy=new_policy,
                                                tokenizer=tokenizer,
                                                reward_fn=reward_fn,
                                                advantage_eps=advantage_eps,
                                                micro_batch_size=eval_batch_size,
                                                group_size=1,
                                                use_std_normalization=use_std_normalization,
                                                off_policy=False)
        output = run_get_response_log_probs(model=new_policy,
                                            input_ids=rollout["input_ids"],
                                            labels=rollout["labels"])
        loss, loss_metadata = run_compute_policy_gradient_loss(
            policy_log_probs=output["log_probs"],
            loss_type=loss_type,
            raw_rewards=rollout["raw_rewards"],
            advantages=rollout["advantages"],
            old_log_probs=rollout["old_log_probs"],
            cliprange=cliprange
        )
        loss_per_example = run_masked_mean(loss, rollout['response_mask'], dim=1)  # (batch_size,)
        loss = loss_per_example.mean()
        total_loss += loss.item()
    return total_loss, metadata
        


def train(experiment,
          train_loader: DataLoader,
          eval_loader: DataLoader,
          tokenizer: PreTrainedTokenizerBase,
          old_policy: LLM,
          new_policy: PreTrainedModel,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
          warmup_scheduler: torch.optim.lr_scheduler._LRScheduler,
          reward_fn: Callable,
          warmup_steps: int,
          n_grpo_steps: int = 200,
          advantage_eps: float = 1e-6,
          rollout_batch_size: int = 256,
          group_size: int = 8,
          epochs_per_rollout_batch: int = 1, # On-policy
          train_batch_size: int = 256, # On-policy
          gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
          loss_type: Literal[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_clip",
          ] = "reinforce_with_baseline",
          cliprange: float = 0.2,
          log_interval: int = 1,
          eval_interval: int = 10,
          use_std_normalization: bool = True,
          output_dir: str = "./output"):
    """
    On-policy setting:
        for each rollout batch, we take a single gradient step, this means that train_batch_size is equal to 
        rollout_batch_size, and epochs_per_rollout_batch is equal to 1.
    Off-policy setting:
        • You should be able to take multiple epochs of gradient steps per rollout batch, where the number of epochs
        and optimizer updates per rollout batch are controlled by rollout_batch_size, epochs_per_rollout_batch, and 
        train_batch_size.
        • Edit your main training loop to get response logprobs from the policy after each rollout batch generation
        phase and before the inner loop of gradient steps—these will be the old_log_probs.
        We suggest using torch.inference_mode().
    And here are a few additional tips:
        • Remember to use the r1_zero pro
        • You should use the "GRPO-Clip" loss type.
        • Remember to use the r1_zero prompt, and direct vLLM to stop generation at the second answer tag
        </answer>, as in the previous experiments.
        • We suggest using typer for argument parsing.
        • Use gradient clipping with clip value 1.0.
        • You should routinely log validation rewards (e.g., every 5 or 10 steps). You should evaluate on at least
        1024 validation examples to compare hyperparameters, as CoT/RL evaluations can be noisy.
        • With our implementation of the losses, GRPO-Clip should only be used when off-policy (since it
        requires the old log-probabilities).
        • In the off-policy setting with multiple epochs of gradient updates per rollout batch, it would be wasteful
        to recompute the old log-probabilities for each epoch. Instead, we can compute the old log-probabilities
        once and reuse them for each epoch.
        • You should not differentiate with respect to the old log-probabilities.
        • You should log some or all of the following for each optimizer update:
            – The loss.
            – Gradient norm.
            – Token entropy.
            – Clip fraction, if off-policy.
            – Train rewards (total, format, and answer).
            – Anything else you think could be useful for debugging.
    """
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    # steps_per_generation in GRPOTrainer, equals to gradient_accumulation_steps when rollout_batch_size is equal to train_batch_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    buffered_inputs = None
    off_policy = epochs_per_rollout_batch > 1 or n_microbatches_per_rollout_batch > gradient_accumulation_steps
    remainder = n_microbatches_per_rollout_batch % gradient_accumulation_steps
    update_steps_in_grpo_step = n_microbatches_per_rollout_batch // gradient_accumulation_steps + int(remainder > 0)
    steps_in_grpo_step = epochs_per_rollout_batch * n_microbatches_per_rollout_batch
    train_iterator = iter(train_loader)
    best_eval_loss = float('inf')
    update_step = 0
    for step in range(n_grpo_steps):
        iter_step = 0
        # Sample a batch(n_prompts_per_rollout_batch) of questions D_b from D
        batch = next(train_iterator)
        # Sample G outputs {o^{(i)}}_{i=1}^G ∼ π_{θ_{old}}(·|q) for each question q ∈ D_b
        # Including compute advantages and rewards
        logging.info(f"=>GRPO-Step [{step}/{n_grpo_steps}], iter-step {iter_step}: Sampling rollout batch...")
        rollouts, data_metadata = sample_rollout(batch=batch,
                                                 vllm_old_policy=old_policy,
                                                 hf_old_policy=new_policy,
                                                 tokenizer=tokenizer,
                                                 reward_fn=reward_fn,
                                                 advantage_eps=advantage_eps,
                                                 micro_batch_size=micro_train_batch_size,
                                                 group_size=group_size,
                                                 use_std_normalization=use_std_normalization,
                                                 off_policy=off_policy)
        experiment.log_metrics(data_metadata, step=iter_step, prefix='rollout')
        rollouts = shuffle_tensor_dict(rollouts)
        buffered_inputs = split_tensor_dict(rollouts, n_microbatches_per_rollout_batch)
        for _ in range(epochs_per_rollout_batch):
            for in_step in range(update_steps_in_grpo_step):
                # Sample tran_batch_size of samples for updating
                num_batches = gradient_accumulation_steps if in_step != (update_steps_in_grpo_step - 1) else remainder
                start = in_step * gradient_accumulation_steps
                batch_samples = buffered_inputs[start: start + num_batches]
                for i, inputs in enumerate(batch_samples):
                    iter_step += 1
                    do_backward = iter_step % gradient_accumulation_steps == 0 or iter_step == steps_in_grpo_step
                    output = run_get_response_log_probs(model=new_policy,
                                                        input_ids=inputs["input_ids"],
                                                        labels=inputs["labels"],
                                                        return_token_entropy=do_backward,
                                                        return_top_token_entropy=do_backward)
                    loss, loss_metadata = run_grpo_microbatch_train_step(
                        policy_log_probs=output["log_probs"],
                        response_mask=inputs["response_mask"],
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=inputs["raw_rewards"],
                        advantages=inputs["advantages"],
                        old_log_probs=inputs["old_log_probs"],
                        cliprange=cliprange
                    )
                    # update every gradient_accumulation_steps or at the end of the last microbatch
                    if do_backward:
                        #  use gradient clipping with clip value 1.0
                        gnorm = torch.nn.utils.clip_grad_norm_(new_policy.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        update_step += 1
                        if step < warmup_steps:
                            warmup_scheduler.step()
                        else:
                            lr_scheduler.step()
                            
                    if iter_step % log_interval == 0:
                        logging.info(f"=>GRPO-Step [{step}/{n_grpo_steps}], iter-step {iter_step}: Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                        experiment.log_metrics({'train/loss': loss.item(),
                             'train/lr': optimizer.param_groups[0]['lr'],
                             'train/avg_next_token_ce': output['token_entropy'].mean().item(),
                             'train/gradient_norm': gnorm.item(),
                             **loss_metadata}, step=iter_step)
                    if iter_step % eval_interval == 0:
                        logging.info(f"=>GRPO-Step [{step}/{n_grpo_steps}], iter-step {iter_step}: Evaluating...")
                        eval_loss, eval_metadata = evaluate(
                            eval_loader=eval_loader,
                            tokenizer=tokenizer,
                            old_policy=old_policy,
                            new_policy=new_policy,
                            reward_fn=reward_fn,
                            advantage_eps=advantage_eps,
                            eval_batch_size=16,
                            loss_type=loss_type,
                            cliprange=cliprange,
                            use_std_normalization=use_std_normalization
                        )
                        experiment.log_metrics({'eval/loss': eval_loss,
                            **eval_metadata}, step=step)
                        logging.info(f"=>GRPO-Step [{step}/{n_grpo_steps}], iter-step {iter_step}: eval_loss={eval_loss:.4f}")
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            logging.info(f"=>GRPO-Step [{step}/{n_grpo_steps}], iter-step {iter_step}: Best eval_loss updated from {best_eval_loss:.4f} to {eval_loss:.4f}")
                            new_policy.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)                                              
        # Set the old policy model
        load_policy_into_vllm_instance(new_policy, old_policy)
        
        

def main(model_name_or_path: str,
         data_path: str,
         eval_data_path: str,
         instruction: str,
         seed: int = 1234,
         n_grpo_steps: int = 200,
         learning_rate: float = 1e-5,
         rollout_batch_size: int = 256,
         group_size: int = 8,
         epochs_per_rollout_batch: int = 1, # On-policy
         train_batch_size: int = 256, # On-policy
         gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
         loss_type: Literal[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_clip",
         ] = "reinforce_with_baseline",
         cliprange: float = 0.2,
         use_std_normalization: bool = True,
         log_interval: int = 1,
         eval_interval: int = 10,
         output_dir: str = "/data/lanyun/worksapce/assignment5-alignment/models/rl"):
    """
    args: map to GRPOConfig
        train_batch_size: total_train_batch_size in GRPOConfig, equals 
            _train_batch_size(micro_train_batch_size here) * gradient_accumulation_steps
        rollout_batch_size: generation_batch_size in GRPOConfig, equals
            _train_batch_size(micro_train_batch_size here) * steps_per_generation
            steps_per_generation is n_microbatches_per_rollout_batch here
            for on-policy setting
                steps_per_generation = gradient_accumulation_steps
                i.e. rollout_batch_size = train_batch_size
            for off-policy setting
                steps_per_generation = n * gradient_accumulation_steps
                i.e. rollout_batch_size > train_batch_size
        group_size: num_generations in GRPOConfig
        epochs_per_rollout_batch: num_iterations in GRPOConfig, also 𝜇 in the GRPO paper.
            for on-policy setting
                epochs_per_rollout_batch = 1
            for off-policy setting
                epochs_per_rollout_batch > 1
    """
    if epochs_per_rollout_batch == 1 and loss_type == "grpo_clip":
        raise ValueError("GRPO-Clip loss type should only be used with off-policy setting (epochs_per_rollout_batch > 1)")
    assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    # steps_per_generation in GRPOTrainer, equals to gradient_accumulation_steps when rollout_batch_size is equal to train_batch_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    if epochs_per_rollout_batch == 1 and loss_type == "grpo_clip":
        raise ValueError("GRPO-Clip loss type should only be used with off-policy setting (epochs_per_rollout_batch > 1)")
    
    if os.path.isfile(model_name_or_path):
        model_name = os.path.splitext(os.path.basename(model_name_or_path))[0]
    else:
        model_name = model_name_or_path
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    vllm_set_random_seed(seed)
    set_logger(log_path=os.path.join(output_dir, "train.log"))
    
    # prepare data
    logging.info("Preparing data...".center(100, "="))
    train_loader = get_data_loader(data_path,
                                   batch_size=n_prompts_per_rollout_batch,
                                   instruction=instruction,
                                   shuffle=True)
    eval_loader = get_data_loader(eval_data_path,
                                  batch_size=16,
                                  shuffle=False)
    
    # initialize model
    logging.info("Initializing model...".center(100, "="))
    policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    old_policy = init_vllm(model_id=model_name_or_path,
                           device="cuda:1")
    
    # initialize optimizer and scheduler
    logging.info("Initializing optimizer and scheduler...".center(100, "="))
    steps_per_grpo_step = epochs_per_rollout_batch * n_microbatches_per_rollout_batch
    update_steps_per_grpo_step = steps_per_grpo_step // gradient_accumulation_steps + int(steps_per_grpo_step % gradient_accumulation_steps > 0)
    total_update_steps = n_grpo_steps * update_steps_per_grpo_step
    warmup_update_steps = int(0.03 * total_update_steps)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_update_steps - warmup_update_steps)
    def linear_warmup(step):
        if step < warmup_update_steps:
            return (step + 1) / warmup_update_steps
        else:
            return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup)
    
    comet.login()
    experiment = comet.start(project_name="SFT",
                             workspace="leg-end",
                             api_key="SJASztLoOjQpW2Sakl2PDV4YZ")
    experiment.log_parameters({
        "learning_rate": 2e-5,
        "weight_decay": 1e-2,
        "train_batch_size": train_batch_size,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "epochs_per_rollout_batch": epochs_per_rollout_batch,
        "loss_type": loss_type,
        "cliprange": cliprange,
        "use_std_normalization": use_std_normalization,
        "log_interval": log_interval,
        "eval_interval": eval_interval,
        "model_name_or_path": model_name_or_path,
        "data_path": data_path,
        "eval_data_path": eval_data_path,
        "instruction": instruction,
        "seed": seed,
    })
    logging.info("Training start...".center(100, "="))
    train(experiment=experiment,
          train_loader=train_loader,
          eval_loader=eval_loader,
          tokenizer=tokenizer,
          old_policy=old_policy,
          new_policy=policy,
          optimizer=optimizer,
          lr_scheduler=cosine_scheduler,
          warmup_scheduler=warmup_scheduler,
          reward_fn=gsm8k_reward_fn,
          warmup_steps=warmup_update_steps,
          n_grpo_steps=n_grpo_steps,
          rollout_batch_size=rollout_batch_size,
          group_size=group_size,
          epochs_per_rollout_batch=epochs_per_rollout_batch,
          train_batch_size=train_batch_size,
          gradient_accumulation_steps=gradient_accumulation_steps,
          loss_type=loss_type,
          cliprange=cliprange,
          use_std_normalization=use_std_normalization,
          log_interval=log_interval,
          eval_interval=eval_interval,
          output_dir=output_dir)
    experiment.end()
    

if __name__ == "__main__":
    main(model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
         data_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/train.jsonl",
         eval_data_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test.jsonl",
         instruction="/data/lanyun/worksapce/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt",
         seed=1234,
         log_interval=10,
         eval_interval=100,
         output_dir="/data/lanyun/worksapce/assignment5-alignment/models/rl")