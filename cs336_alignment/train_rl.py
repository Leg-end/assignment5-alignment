import os
from typing import Literal, Callable
from vllm import SamplingParams, LLM, RequestOutput
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.train_sft import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.utils import set_logger
from torch.utils.data import DataLoader
from tests.adapters import run_compute_group_normalized_rewards, run_grpo_microbatch_train_step,\
    run_get_response_log_probs, run_tokenize_prompt_and_output
from datasets import load_dataset

import torch
import logging


def train(data_loader: DataLoader,
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
          total_steps: int = 1000,
          log_interval: int = 1,
          eval_interval: int = 10,
          use_std_normalization: bool = True):
    """
    And here are a few additional tips:
    • Remember to use the r1_zero prompt, and direct vLLM to stop generation at the second answer tag
    </answer>, as in the previous experiments.
    • Wesuggest using typer for argument parsing.
    • Use gradient clipping with clip value 1.0.
    • Youshould routinely log validation rewards (e.g., every 5 or 10 steps). You should evaluate on at least
    1024 validation examples to compare hyperparameters, as CoT/RL evaluations can be noisy.
    • With our implementation of the losses, GRPO-Clip should only be used when off-policy (since it
    requires the old log-probabilities).
    • Intheoff-policy setting with multiple epochs of gradient updates per rollout batch, it would be wasteful
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
    while step < total_steps:
        for _ in range(n_grpo_steps):
            # Sample a n_prompts_per_rollout_batch of questions D_b from D
            prompt_strs, answer_strs = next(data_loader)
            # Set the old policy model
            load_policy_into_vllm_instance(new_policy, old_policy)
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
            # TODO filter out wrong outputs with low rewards
            batch = run_tokenize_prompt_and_output(prompt_strs, answer_strs, tokenizer)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            for _ in range(epochs_per_rollout_batch):  # off-policy if epochs_per_rollout_batch > 1
                do_backward = (step + 1) % gradient_accumulation_steps == 0
                output = run_get_response_log_probs(model=new_policy,
                                                    input_ids=input_ids,
                                                    labels=labels,
                                                    return_token_entropy=do_backward,
                                                    return_token_id=do_backward)
                loss, metadata = run_grpo_microbatch_train_step(
                    policy_log_probs=output["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=raw_rewards,
                    advantages=advantages,
                    old_log_probs=old_log_probs,
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