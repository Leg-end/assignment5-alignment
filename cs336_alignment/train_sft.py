import os
os.environ["HF_ENDPOINT"] = 'https:/hf-mirror.com'
os.environ["HF_HOME"] = "/data/lanyun/worksapce/assignment5-alignment/models"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams, RequestOutput
from unittest.mock import patch
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from tests.adapters import run_get_response_log_probs, run_sft_microbatch_train_step, run_masked_normalize, run_tokenize_prompt_and_output
from cs336_alignment.data_utils import get_data_loader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
from cs336_alignment.utils import set_logger
from cs336_alignment.evaluate import evaluate_vllm
from typing import Callable
from collections import defaultdict
import logging

import torch
import wandb


def init_vllm(model_id: str,
              device: str,
              gpu_memory_utilization: float = 0.85) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # vLLM is not compatible with accelerate. So we need to patch it to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    In order to track the progress of your model over the course of training, you should periodically evaluate
    it on the MATH validation set. You should run your script with 2 GPUs, using one GPU for the policy
    model and the other for the vLLM instance to evaluate the policy.
    
    load the policy weights into the vLLM instance for evaluation.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    

@torch.no_grad()
def evaluate(eval_loader: DataLoader,
             model: LLM,
             sampling_params: SamplingParams):
    prompt_strs = eval_loader.dataset.prompts
    response_strs = eval_loader.dataset.answers
    eval_results = evaluate_vllm(vllm_model=model,
                                prompts=prompt_strs,
                                gt_answers=response_strs,
                                eval_sampling_params=sampling_params)
    eval_loss = eval_results['eval/total_loss']
    eval_loss /= len(prompt_strs)
    return eval_loss, eval_results
    

def train(run,
          train_loader: DataLoader,
          eval_loader: DataLoader,
          tokenizer: PreTrainedTokenizerBase,
          model: PreTrainedModel,
          eval_model: LLM,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
          warmup_scheduler: torch.optim.lr_scheduler._LRScheduler,
          total_steps: int,
          warmup_steps: int,
          gradient_accumulation_steps: int = 8,
          log_interval: int = 100,
          eval_interval: int = 1000,
          output_dir: str = "./output"):
    """
    Input: initial policy model; SFT Dateset D
    for step = 1, ..., n_sft_steps do
        Sample a batch of question-response pairs D_b from D
        Compute the cross-entropy loss of the responses given the questions using the model
        Update the model parameters θ by taking a gradient step with respect to the cross-entropy loss
    end for
    Output: trained policy model
    """

    model.train()
    device = model.device
    step = 0
    best_eval_loss = float('inf')
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
                                     include_stop_str_in_output=True, logprobs=1)
    tokenizer = None
    while step < total_steps:
        for idx, batch in enumerate(train_loader):
            prompt_strs, response_strs = batch
            batch = run_tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            do_backward = (idx + 1) % gradient_accumulation_steps == 0
            output = run_get_response_log_probs(model=model,
                                                input_ids=input_ids,
                                                labels=labels,
                                                return_token_entropy=do_backward,
                                                return_token_id=do_backward)
            loss, metadata = run_sft_microbatch_train_step(
                policy_log_probs=output["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
                return_metadata=do_backward
            )
            #  use gradient clipping with clip value 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if do_backward:
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                if step < warmup_steps:
                    warmup_scheduler.step()
                else:
                    lr_scheduler.step()
                
                if step % log_interval == 0:
                    logging.info(f"Step [{step}/{total_steps}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                    run.log({'train/loss': loss.item(),
                             'train/lr': optimizer.param_groups[0]['lr'],
                             'train/avg_next_token_ce': output['token_entropy'].mean().item(),
                             **metadata}, step=step)
                    
                if step % eval_interval == 0:
                    logging.info("Evaluation start...".center(100, "="))
                    load_policy_into_vllm_instance(model, eval_model)
                    eval_loss, eval_metadata = evaluate(eval_loader, eval_model, sampling_params)
                    logging.info(f"Step [{step}/{total_steps}], Eval Loss: {eval_loss:.4f}")
                    run.log({'eval/loss': eval_loss,
                             **eval_metadata}, step=step)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        model.save_pretrained(save_directory=output_dir)
                        tokenizer.save_pretrained(save_directory=output_dir)


def main(model_name_or_path: str,
         data_path: str,
         instruction: str,
         gradient_accumulation_steps: int,
         eval_data_path: str | None = None,
         num_examples: int = 128,
         total_steps: int = 6200,
         seed: int = 1234,
         log_interval: int = 100,
         eval_interval: int = 1000,
         output_dir: str="/data/lanyun/worksapce/assignment5-alignment/models/sft"):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_loader, eval_loader = get_data_loader(data_path=data_path,
                                                eval_data_path=eval_data_path,
                                                 instruction=instruction,
                                                 num_examples=num_examples,
                                                 batch_size=8,
                                                 num_workers=4,
                                                 eval_ratio=0.1,
                                                 eval_batch_size=2)
    # initialize model
    logging.info("Initializing model...".center(100, "="))
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to("cuda:0")
    eval_model = init_vllm(model_id=model_name_or_path,
                           device="cuda:1")
    # initialize optimizer and scheduler
    logging.info("Initializing optimizer and scheduler...".center(100, "="))
    warmup_steps = int(0.03 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    def linear_warmup(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup)
    
    wandb.login()
    with wandb.init(
            project=f"SFT",
            name=f"{model_name}-data_size-{num_examples}",
            config={
                "learning_rate": 2e-5,
                "weight_decay": 1e-2,
                "train_batch_size": 8,
                "eval_batch_size": 2,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            mode="offline"
        ) as run:
        run.define_metric("train_step")
        run.define_metric("eval_step")
        run.define_metric("train/*", step_metric="train_step")
        run.define_metric("eval/*", step_metric="eval_step")
        logging.info("Training start...".center(100, "="))
        logging.info(f"Total epochs around {total_steps // len(train_loader) // 4}")
        train(run,
              train_loader=train_loader,
            eval_loader=eval_loader,
            model=model,
            eval_model=eval_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=cosine_scheduler,
            warmup_scheduler=warmup_scheduler,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_interval=log_interval,
            eval_interval=eval_interval,
            output_dir=output_dir)
    

if __name__ == "__main__":
    main(model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
         data_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/train.jsonl",
         eval_data_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test.jsonl",
         instruction="/data/lanyun/worksapce/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt",
         gradient_accumulation_steps=8,
         total_steps=6200,
         seed=1234,
         num_examples=128,
         log_interval=10,
         eval_interval=100,
         output_dir="/data/lanyun/worksapce/assignment5-alignment/models/sft")
    