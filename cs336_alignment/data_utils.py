import os
# os.environ["HF_ENDPOINT"] = 'https:/hf-mirror.com'
# os.environ["HF_HOME"] = "/data/lanyun/worksapce/assignment5-alignment/models"
from xopen import xopen
import json
import pandas as pd
from typing import Any
from torch.utils.data import Dataset, DataLoader, random_split, Subset


def look_data(data_path: str, num_samples: int = 10) -> None:
    with open(data_path, "r") as f, open(data_path.replace(".json", "_sample.json"), "w") as fw:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            fw.write(json.dumps(data, indent=4) + "\n")


def may_load_instruction(instruction: str):
    if os.path.isfile(instruction) and os.path.exists(instruction):
        instruction = xopen(instruction, "r", encoding="utf-8").read()
    return instruction    


def alpaca_data_loader(data_path: str,
                       instruction: str,
                       ):
    instruction = may_load_instruction(instruction)
    prompts = []
    answers = []
    with xopen(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(instruction.format(instruction=data["instruction"]))
            answers.append(data["output"])
    return prompts, answers


def gsm8k_data_loader(data_path: str,
                      instruction: str):
    instruction = may_load_instruction(instruction)
    prompts = []
    answers = []
    with xopen(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(instruction.format(question=data["question"]))
            answer = data["answer"]
            think, answer = answer.rsplit('\n', maxsplit=1)
            answer = answer[4:]
            response = f" {think} </think> <answer>{answer} </answer>"
            answers.append(response)
    return prompts, answers


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
    # raise NotImplementedError
    legal_options = ["A", "B", "C", "D"]
    
    # model_option = 
    for option in legal_options:
        if option in model_output:
            return option
    
    # If no option is found, return None
    return None


def mmlu_data_loader(data_path: str,
                     instruction: str):
    instruction = may_load_instruction(instruction)
    df_mmlu = pd.DataFrame(columns=['Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D', 'Answer', 'tag', 'split'])

    for split in ['dev', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        for file_name in os.listdir(split_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(split_path, file_name)
                df = pd.read_csv(file_path, header=None)
                df.columns = ['Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D', 'Answer']

                df['tag'] = file_name.replace(f'_{split}.csv', '')
                df['split'] = split
                
                df_mmlu = pd.concat([df_mmlu, df], ignore_index=True)
    prompts = []
    answers = []
    for index, row in df_mmlu.iterrows():
        if not row['split'] == 'test':
            continue
        
        question = row['Question']
        options = [row['Option_A'], row['Option_B'], row['Option_C'], row['Option_D']]
        answer = row['Answer']
        subject = row['tag']
        prompt = instruction.format(subject=subject, question=question, option1=options[0],
                                    option2=options[1], option3=options[2], option4=options[3])
        prompts.append(prompt)
        answers.append(answer)
        

def mmlu_reward_fn(response, ground_truth):
    parsed_answer = run_parse_mmlu_response(None, response)
    if parsed_answer == ground_truth:
        reward = 1.0
    else:
        reward = 0.0
    return {"parsed_answer": parsed_answer, "reward": reward}


class SFTDataset(Dataset):
    """
     a formatted prompt and a target response, where the target response includes a chain-of-thought
     reasoning trace and the final answer
    """
    def __init__(self,
                 data_path: str,
                 instruction: str):
        super().__init__()
        self.prompts, self.answers = gsm8k_data_loader(data_path, instruction)
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        prompt = self.prompts[index]
        answer = self.answers[index]
        # think, answer = answer.rsplit('\n', maxsplit=1)
        # answer = answer[4:]
        # response = f" {think} </think> <answer>{answer} </answer>"
        return prompt, answer
    

def get_data_loader(data_path: str,
                    instruction: str,
                    batch_size: int,
                    num_workers: int,
                    eval_data_path: str | None = None,
                    num_examples: int | None = None,
                    eval_ratio: float = 0.0,
                    eval_batch_size: int | None = None):
    # def collate_fn(batch):
    #     prompt_strs = []
    #     output_strs = []
    #     for prompt, answer in batch:
    #         prompt_strs.append(prompt)
    #         output_strs.append(answer)
    #     return run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    
    dataset = SFTDataset(data_path, instruction)
    if eval_data_path is None and eval_ratio == 0.0:
        if num_examples is not None:
            num_examples = min(max(128, num_examples), len(dataset))
            dataset = Subset(dataset, range(num_examples))
        data_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        return data_loader
    elif eval_data_path is not None:
        if num_examples is not None:
            num_examples = min(max(128, num_examples), len(dataset))
            dataset = Subset(dataset, range(num_examples))
        train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        eval_dataset = SFTDataset(eval_data_path, instruction)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=eval_batch_size or batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
        return train_loader, eval_loader
    elif eval_ratio > 0.0:
        eval_ratio = max(min(eval_ratio, 0.3), 0.1)
        eval_size = int(len(dataset) * eval_ratio)
        eval_batch_size = eval_batch_size or batch_size
        train_dataset, eval_dataset = random_split(dataset, [len(dataset) - eval_size, eval_size])
        if num_examples is not None:
            num_examples = min(max(128, num_examples), len(train_dataset))
            train_dataset = Subset(train_dataset, range(num_examples))
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
        return train_loader, eval_loader


if __name__ == '__main__':
    data_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/train.jsonl"
    eval_data_path="/data/lanyun/worksapce/assignment5-alignment/data/gsm8k/test.jsonl"
    instruction="/data/lanyun/worksapce/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    train_loader, eval_loader = get_data_loader(data_path=data_path,
                                                eval_data_path=eval_data_path,
                                                 instruction=instruction,
                                                 num_examples=128,
                                                 batch_size=4,
                                                 num_workers=4,
                                                 eval_ratio=0.1,
                                                 eval_batch_size=2)
    for i, batch in enumerate(train_loader):
        a, b = batch
        print(a)
        print(b)
        break