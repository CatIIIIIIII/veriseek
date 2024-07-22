import re
from datasets import load_dataset
from tqdm import tqdm
import os
from transformers import AutoTokenizer


def redact_email(text):
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return email_pattern.sub("", text)

def redact_phone(text):
    phone_pattern = re.compile(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}\b|\b\d{3}[-.\s]??\d{4}\b')
    return phone_pattern.sub("", text)

def redact_ssn(text):
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    return ssn_pattern.sub("", text)

def redact_text(text):
    text = redact_email(text)
    text = redact_phone(text)
    text = redact_ssn(text)
    return text


data_path = "/data/raw/new_GPU_SIM_NEW_V"
data_files = []
# find all .v file and iterate it 
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".v"):
            data_files.append(os.path.join(root, file))

dataset = []
data_length = []
ori_len = len(data_files)
with tqdm(len(data_files)) as pbar:
    for df in data_files:
        pbar.update(1)
        with open(df, 'r') as f:
            # read f as a whole string
            text = f.read()
            # redact text
            modified_text = redact_text(text)
            length = len(modified_text.split())
            if length < 5000:
                dataset.append({'text': modified_text})
                data_length.append(length)
    
tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/codellama-7B/finetune/best/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )
dataset_short = []
with tqdm(len(dataset)) as pbar:
    for data in dataset:
        lens = len(tokenizer(data["text"])["input_ids"])
        if lens < 4096:
            dataset_short.append({'text': data["text"]})
        pbar.update(1)

# save data_raw_ as new json file
import json
with open('gpu_sim.jsonl', 'w') as f:
    json.dump(dataset_short, f)

print(f"Original length: {ori_len}, After filtering: {len(dataset)}. Take {len(dataset)/ori_len*100:.2f}% of the original data.")

# plot the distribution of the length of the text
import matplotlib.pyplot as plt
plt.hist(data_length, bins=100)
plt.savefig('gpu_sim_length_distribution.png')

from datasets import load_dataset
ds = load_dataset("json", data_files="gpu_sim.jsonl")
print(len(ds["train"]))