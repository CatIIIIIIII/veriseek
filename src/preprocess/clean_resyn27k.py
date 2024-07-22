import json
from tqdm import tqdm
from transformers import AutoTokenizer

with open('Resyn27k.json', 'r') as json_file:
    json_list = list(json_file)

tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/codellama-7B/finetune/best/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )

length = []
dataset = []
dataset_pretrain = []
with tqdm(len(json_list)) as pbar:
    for json_str in json_list:
        data = json.loads(json_str)
        lens = len(tokenizer(data["Instruction"] + "\n" + data['Response'][0])["input_ids"])
        length.append(lens)
        if lens < 2048:
            dataset.append({
                'instruction': data['Instruction'],
                'input': None,
                'output': data['Response'][0]
            })
            dataset_pretrain.append({'text': data['Response'][0]})
        pbar.update(1)

with open('instruct_resyn.jsonl', 'w') as f:
    json.dump(dataset, f)

with open('resyn_pretrain.jsonl', 'w') as f:
    json.dump(dataset_pretrain, f)
    
# plot the distribution of the length of the text
import matplotlib.pyplot as plt
plt.hist(length, bins=100)
plt.savefig('resyn_text_length_distribution.png')

from datasets import load_dataset
ds = load_dataset("json", data_files="instruct_resyn.jsonl")
print(len(ds["train"]))