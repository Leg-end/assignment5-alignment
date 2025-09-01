import os
from typing import Literal, Callable
from vllm import SamplingParams, LLM, RequestOutput
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.train_sft import init_vllm, load_policy_into_vllm_instance, evaluate
from cs336_alignment.utils import set_logger
from torch.utils.data import DataLoader
from tests.adapters import run_compute_group_normalized_rewards, run_grpo_microbatch_train_step,\
    run_get_response_log_probs, run_tokenize_prompt_and_output
from datasets import load_dataset

import torch
import logging


@torch.inference_mode()
def sample_rollout(train_loader: DataLoader,
                   old_policy: LLM,
                   sampling_params: SamplingParams,
                   reward_fn: Callable,
                   device: str,
                   advantage_eps: float = 1e-6,
                   group_size: int = 8,
                   use_std_normalization: bool = True) -> tuple[dict[str, torch.Tensor | list[str]], dict[str, float]]:
    """
    Sample a batch of trajectories from the current policy.
    Args:
        train_loader: DataLoader for training data
        old_policy: the current policy used to sample trajectories
        reward_fn: function to compute reward for each trajectory
        group_size: number of trajectories to sample per prompt
    Returns:
        prompt_strs: list of prompt strings
        rollout_responses: list of sampled responses
        repeated_ground_truths: list of ground truth answers repeated for each response
        old_log_probs: list of log probabilities of the sampled responses
        advantages: tensor of advantages for each response
        raw_rewards: tensor of raw rewards for each response
        metadata: additional metadata from reward function
    """
    # Sample a n_prompts_per_rollout_batch of questions D_b from D
    prompt_strs, answer_strs = next(train_loader)
    # Sample group_size response for each question in the batch, rollout_batch_size = n_prompts_per_rollout_batch * group_size
    request_outputs: list[RequestOutput] = old_policy.generate(prompt_strs, sampling_params)
    rollout_responses = []
    repeated_ground_truths = []
    old_log_probs = []
    for idx, request_output in enumerate(request_outputs):
        answer_str = answer_strs[idx]
        for output in request_output.outputs:  # group size of response for a prompt
            response = output.text.strip()
            rollout_responses.append(response)
            repeated_ground_truths.append(answer_str)
            log_probs = [logprob[token_id].logprob for token_id, logprob in zip(output.token_ids, output.logprobs)]
            old_log_probs.append(log_probs)
    # will be reused on off-policy
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
    # Compute rewards for each sampled response by running reward function
    # and advantages with group normalization
    advantages, raw_rewards, metadata = run_compute_group_normalized_rewards(
        reward_fn=reward_fn, rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size, advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization
    )
    return {"prompt_strs": prompt_strs,
            "answer_strs": answer_strs,
            "rollout_responses": rollout_responses,
            "repeated_ground_truths": repeated_ground_truths,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "raw_rewards": raw_rewards}, metadata


def train(experiment,
          train_loader: DataLoader,
          eval_loader: DataLoader,
          tokenizer: PreTrainedTokenizerBase,
          old_policy: LLM,
          new_policy: PreTrainedModel,
          optimizer: torch.optim.Optimizer,
          reward_fn: Callable,
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
          use_std_normalization: bool = True):
    """
    args:
        rollout_batch_size: number of trajectories when exploration, equal to train_batch_size in on-policy setting,
            larger in off-policy setting.
        train_batch_size:  number of samples sampled from replay buffer for updating the policy, equal to
            rollout_batch_size in on-policy setting
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
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    sampling_params = SamplingParams(n=group_size,
                                     seed=None,  # ensure diversity
                                     temperature=1.0,
                                     top_p=1.0,
                                     min_tokens=4, # As in Expiter, disallow empty string responses
                                     max_tokens=1024,
                                     stop=['</answer>'],
                                     include_stop_str_in_output=True)
    device = new_policy.device
    step = 0
    while step < n_grpo_steps:
        rollouts, metadata = sample_rollout(train_loader=train_loader,
                                            old_policy=old_policy,
                                            sampling_params=sampling_params,
                                            reward_fn=reward_fn,
                                            device=device,
                                            advantage_eps=advantage_eps,
                                            group_size=group_size,
                                            use_std_normalization=use_std_normalization)
        for i in range(0, train_batch_size, micro_train_batch_size):
            micro_rollouts = {k: v[i: i + micro_train_batch_size] for k, v in rollouts.items()}
            # TODO filter out wrong outputs with low rewards
            micro_batch = run_tokenize_prompt_and_output(micro_rollouts["prompt_strs"], micro_rollouts["answer_strs"], tokenizer)
            micro_input_ids = micro_batch["input_ids"].to(device)
            micro_labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            micro_input_ids = input_ids[i: i + micro_train_batch_size]
            micro_labels = labels[i: i + micro_train_batch_size]
            micro_response_mask = response_mask[i: i + micro_train_batch_size]
            micro_raw_rewards = raw_rewards[i: i + micro_train_batch_size]
            micro_advantages = advantages[i: i + micro_train_batch_size]
            micro_old_log_probs = old_log_probs[i: i + micro_train_batch_size]
            for _ in range(epochs_per_rollout_batch):  # off-policy if epochs_per_rollout_batch > 1
                do_backward = (step + 1) % gradient_accumulation_steps == 0
                output = run_get_response_log_probs(model=new_policy,
                                                    input_ids=micro_input_ids,
                                                    labels=micro_labels,
                                                    return_token_entropy=do_backward,
                                                    return_token_id=do_backward)
                loss, metadata = run_grpo_microbatch_train_step(
                    policy_log_probs=output["log_probs"],
                    response_mask=micro_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=micro_raw_rewards,
                    advantages=micro_advantages,
                    old_log_probs=micro_old_log_probs,
                    cliprange=cliprange,
                )
                if do_backward:
                    #  use gradient clipping with clip value 1.0
                    torch.nn.utils.clip_grad_norm_(new_policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    step += 1
                # log training stats
                if step % log_interval == 0:
                    logging.info(f"Step {step}: loss={loss.item():.4f}")
                    for k, v in metadata.items():
                        if isinstance(v, float):
                            logging.info(f"{k}={v:.4f}")
                        else:
                            logging.info(f"{k}={v}")
                            
                if step % eval_interval == 0:
                    logging.info()
        # Set the old policy model
        load_policy_into_vllm_instance(new_policy, old_policy)
        
        

def main(model_name_or_path: str,
         seed: int = 1234,
         learning_rate: float = 1e-5,
         log_interval: int = 1,
         eval_interval: int = 10,
         output_dir: str = "/data/lanyun/worksapce/assignment5-alignment/models/rl"):
    if os.path.isfile(model_name_or_path):
        model_name = os.path.splitext(os.path.basename(model_name_or_path))[0]
    else:
        model_name = model_name_or_path
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    vllm_set_random_seed(seed)
    set_logger(log_path=os.path.join(output_dir, "train.log"))
    
    # initialize model
    logging.info("Initializing model...".center(100, "="))
    policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy.to("cuda:0")
    old_policy = init_vllm(model_id=model_name_or_path,
                           device="cuda:1")
    
    # initialize optimizer and scheduler
    logging.info("Initializing optimizer and scheduler...".center(100, "="))
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    logging.info("Training start...".center(100, "="))
    train(old_policy,
          policy,
          optimizer)