from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings separately, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    input_ids_list = []
    response_mask_list = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
        output_ids = tokenizer(output_str, add_special_tokens=False)['input_ids']
        input_ids = prompt_ids + output_ids
        response_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        response_mask_list.append(torch.tensor(response_mask, dtype=torch.long))
    
    max_len = max(len(ids) for ids in input_ids_list)
    batch_input_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
    batch_reponse_masks = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    # padding on the right side, since it's for training
    for i, (input_ids, response_mask) in enumerate(zip(input_ids_list, response_mask_list)):
        seq_len = len(input_ids)
        batch_input_ids[i, :seq_len] = input_ids
        batch_reponse_masks[i, :seq_len] = response_mask
    return {
        "input_ids": batch_input_ids[:, :-1],
        "labels": batch_input_ids[:, 1:],
        "response_mask": batch_reponse_masks[:, 1:],
    }   


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = []
    format_rewards = []
    answer_rewards = []
    # for a batch of questions, each question will samples group_size response
    # that makes total batch_size x group_size responses
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        rewards.append(reward_dict["reward"])
        format_rewards.append(reward_dict["format_reward"])
        answer_rewards.append(reward_dict["answer_reward"])
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    N = len(rewards)
    assert N % group_size == 0, "Rollout batch size must be divisible by group_size"
    n_groups = N // group_size
    
    group_rewards = rewards.view(n_groups, group_size)
    group_means = group_rewards.mean(dim=-1, keepdim=True)
    normalized_rewards = group_rewards - group_means
    if normalize_by_std:
        group_stds = group_rewards.std(dim=-1, keepdim=True)
        normalized_rewards = normalized_rewards / (group_stds + advantage_eps)
    normalized_rewards = normalized_rewards.view(-1)
    
    metadata = {
        "reward/mean": rewards.mean().item(),
        "reward/std": rewards.std().item(),
        "reward/max": rewards.max().item(),
        "reward/min": rewards.min().item(),
        "format_reward/mean": sum(format_rewards) / N,
        "answer_reward/mean": sum(answer_rewards) / N
    }
    return normalized_rewards, rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # - sum(p(x) * log(p(x))) = sum(p(x) * logsumexp(l(x))) - sum(p(x) * l(x)) = logsumexp(l(x)) - sum(p(x) * l(x))
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    expected_logits = torch.sum(probs * logits, dim=-1)
    entropy = lse - expected_logits
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
    return_top_token_entropy: bool = False,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits  # batch, seq_len, vocab_size
    log_probs = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return_dict = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = run_compute_entropy(logits)
        return_dict["token_entropy"] = token_entropy
    if return_top_token_entropy:
        top_logits, top_token_ids = torch.max(logits, dim=-1)
        return_dict["top_token_ids"] = top_token_ids
        return_dict["top_log_probs"] = top_logits - torch.logsumexp(logits, dim=-1)
    return return_dict


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)  # importance sampling ratio
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    lhs, rhs = ratio * advantages, clipped_ratio * advantages
    loss = -torch.min(lhs, rhs)
    
    metadata = {
        "clipped": (rhs < lhs).float()
    }
    return loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    batch_size, seq_len = policy_log_probs.shape
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for loss_type=no_baseline"
        assert raw_rewards.shape == (batch_size, 1), "raw_rewards must have shape (batch_size, 1)"
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for loss_type=reinforce_with_baseline"
        assert advantages.shape == (batch_size, 1), "advantages must have shape (batch_size, 1)"
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for loss_type=grpo_clip"
        assert advantages.shape == (batch_size, 1), "advantages must have shape (batch_size, 1)"
        assert cliprange is not None, "cliprange must be provided for loss_type=grpo_clip"
        assert old_log_probs is not None, "old_log_probs must be provided for loss_type=grpo_clip"
        assert old_log_probs.shape == (batch_size, seq_len), "old_log_probs must have shape (batch_size, sequence_length)"
        loss, metadata = run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata['clip_fraction'] = metadata['clipped'].sum().item() / metadata['clipped'].numel()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss, metadata


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    n_tokens = mask.sum(dim=dim)
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / n_tokens


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
    return_metadata: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size, seq_len = policy_log_probs.shape
    total_loss = run_masked_normalize(-policy_log_probs, response_mask, normalize_constant=normalize_constant)
    loss = total_loss / batch_size / gradient_accumulation_steps
    loss.backward()
    
    if return_metadata:
        n_tokens = response_mask.sum().item()
        avg_token_ce = total_loss.item() / (n_tokens + 1e-8)
        metadata = {"train/avg_token_ce": avg_token_ce,
                    "train/total_loss": total_loss.item(),
                    "train/n_tokens": n_tokens}
    else:
        metadata = {}
    return loss.detach(), metadata

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = run_compute_policy_gradient_loss(policy_log_probs=policy_log_probs,
                                                      loss_type=loss_type,
                                                      raw_rewards=raw_rewards,
                                                      advantages=advantages,
                                                      old_log_probs=old_log_probs,
                                                      cliprange=cliprange)
    loss_per_example = run_masked_mean(loss, response_mask, dim=1)  # (batch_size,)
    loss = loss_per_example.mean() / gradient_accumulation_steps
    loss.backward()
    metadata['microbatch/loss'] = loss.detach()
    metadata['microbatch/loss_per_example'] = loss_per_example.detach()
    return loss, metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    sum_val = masked_tensor.sum(dim=dim)
    return sum_val / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
