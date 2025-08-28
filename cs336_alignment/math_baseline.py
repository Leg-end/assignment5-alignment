import os
os.environ["HF_ENDPOINT"] = 'https:/hf-mirror.com'
os.environ["HF_HOME"] = "/data/lanyun/worksapce/assignment5-alignment/models"
from vllm import LLM, SamplingParams
# https://github.com/vllm-project/vllm/blob/main/examples/offline_inference
from typing import Callable
from xopen import xopen
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from collections import defaultdict

import json
import logging
logger = logging.getLogger(__name__)


def parquet_to_jsonl(parquet_path):
    import pandas as pd
    
    df = pd.read_parquet(parquet_path)
    jsonl_path = parquet_path.replace(".parquet", ".jsonl")
    df.to_json(jsonl_path, orient='records', lines=True)


def evaluate_vllm(
    vllm_model: LLM,
    prompts: list[str],
    gt_answers: list[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
    reward_fn: Callable[[str, str], dict[str, float]] = r1_zero_reward_fn,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    vllm_model:
        • Qwen 2.5 Math 1.5B Base (for reasoning experiments):
        /data/a5-alignment/models/Qwen2.5-Math-1.5B
        • Llama 3.1 8B Base (for optional instruction tuning experiments):
        /data/a5-alignment/models/Llama-3.1-8B
        • Llama 3.3 70B Instruct (for optional instruction tuning experiments):
        /data/a5-alignment/models/Llama-3.3-70B-Instruct
    """
    raw_responses = vllm_model.generate(prompts, eval_sampling_params)
    responses = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()
        responses.append(response)
    assert len(responses) == len(prompts)
    logger.info(f"Processed {len(prompts)} prompts")
    
    if reward_fn is not None:
        with xopen(output_path, "w") as f:
            for prompt, response, answer in zip(prompts, responses, gt_answers):
                reward_dict = reward_fn(response, answer)
                result = {"prompt": prompt, "answer": answer, "response": response, **reward_dict}
                f.write(json.dumps(result) + '\n')


def object_to_dict(obj):
    """递归将对象转换为字典"""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj  # 基本类型直接返回
    elif isinstance(obj, (list, tuple, set)):
        return [object_to_dict(item) for item in obj]  # 递归处理列表/元组/集合
    elif isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}  # 递归处理字典
    else:
        # 自定义类：提取属性并递归处理
        result = {}
        for attr, value in obj.__dict__.items():
            if not attr.startswith('_'):  # 忽略私有属性（如 __class__）
                result[attr] = object_to_dict(value)
        return result
    

def main(input_path: str,
         model_name_or_path: str,
         output_path: str | None = None,
         num_samples: int = 10,) -> None:
    if output_path is None:
        output_path = input_path.replace(".jsonl", "_output.jsonl")
    if not os.path.isfile(model_name_or_path):
        assert model_name_or_path in ['Qwen/Qwen2.5-Math-1.5B', 'Llama/Llama-3.1-8B', 'Llama/Llama-3.3-70B-Instruct']
    # (1) load the validation examples, .jsonl format
    questions, answers = [], []
    with open(input_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            answers.append(data['answer'])
            if len(questions) == num_samples:
                break
            
    # (2) format them as string prompts to the language model
    prompts = []
    instruction = open("/data/lanyun/worksapce/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt", 'r', encoding="utf-8").read()
    for question in questions:
        prompts.append(instruction.format(question=question))
    
    # (3) generate outputs for each example
    # (4) calculate evaluation metrics
    # (5) serialize results {examples, model generations, evaluation scores} to disk
    model = LLM(model=model_name_or_path)
    sampling_params = SamplingParams(n=2, temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
                                     include_stop_str_in_output=True, logprobs=1)
    raw_responses = model.generate(prompts, sampling_params)
    raw_responses = object_to_dict(raw_responses)
    json.dump(raw_responses, indent=4, fp=open(output_path, 'w'))
    # evaluate_vllm(vllm_model=model,
    #               reward_fn=r1_zero_reward_fn,
    #               prompts=prompts,
    #               gt_answers=answers,
    #               eval_sampling_params=sampling_params,
    #               output_path=output_path)
    


def analyse_reward(result_path: str):
    reward_cate = defaultdict(int)
    case_at_lease = 10
    reward_0_cases = []
    reward_1_answer_0_cases = []
    with open(result_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            reward_cate[data['format_reward'], data['answer_reward']] += 1
            if data['reward'] == 0 and len(reward_0_cases) < case_at_lease:
                reward_0_cases.append(data)
            if data['reward'] == 1 and data['answer_reward'] == 0 and len(reward_1_answer_0_cases) < case_at_lease:
                reward_1_answer_0_cases.append(data)
    print(f"reward_cate: {reward_cate}")
    json.dump({"reward_0_cases": reward_0_cases, "reward_1_answer_0_cases": reward_1_answer_0_cases}, indent=4, fp=open(result_path.replace(".jsonl", "_analysis.json"), 'w'))
    

def debug_reward_fn(result_path: str):
    case_at_lease = 2
    with open(result_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            answer = data['answer']
            response = data['response']
            reward = r1_zero_reward_fn(response, answer, fast=False)
            print(reward)
            case_at_lease -= 1
            if case_at_lease == 0:
                break
            

def slow_reward_check(result_path: str):
    with open(result_path, 'r', encoding="utf-8") as f, open(result_path.replace(".jsonl", "_slow.jsonl"), 'w') as f_slow:
        for line in f:
            data: dict = json.loads(line)
            answer = data['answer']
            response = data['response']
            reward = r1_zero_reward_fn(response, answer, fast=False)
            data.update(reward)
            f_slow.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    # main(input_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test.jsonl",
    #      model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
    #      output_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test_output_debug.jsonl",
    #      num_samples=2)
    # analyse_reward(result_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test_output_slow.jsonl")
    # slow_reward_check(result_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test_output.jsonl")
    from transformers import AutoTokenizer
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    text = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt},
             {"role": "user", "content": "What is the area of a circle with a radius of 5?"}], tokenize=False, add_generation_prompt=True)
    print(text)