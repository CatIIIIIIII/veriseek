import re
import json

from tqdm import tqdm
from datasets import load_dataset
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


data_files = {"train": "vgen_nodupl.csv"}
data_raw = load_dataset(path="/data/raw", data_files=data_files)["train"]
data_raw_ = []
data_length = []
with tqdm(len(data_raw)) as pbar:
    for data in data_raw:
        pbar.update(1)
        modified_text = redact_text(data["text"])
        length = len(modified_text.split())
        if length < 5000:
            data_length.append(length)
            data_raw_.append({"text": modified_text})

# extract if tokenizer < 2048
tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/codellama-7B/finetune/best/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )
dataset_short = []
with tqdm(len(data_raw_)) as pbar:
    for data in data_raw_:
        lens = len(tokenizer(data["text"])["input_ids"])
        if lens < 4096:
            dataset_short.append({'text': data["text"]})
        pbar.update(1)

dataset_invalid = [{'text': t['text']} for t in dataset_short if 'sky130_' in t['text']]
with open('vgen_nodupl_noiop_invalid.jsonl', 'w') as f:
    json.dump(dataset_invalid, f)
# save data_raw_ as new json file
with open('vgen_nodupl_noiop.jsonl', 'w') as f:
    json.dump(dataset_short, f)

# plot the distribution of the length of the text
import matplotlib.pyplot as plt
plt.hist(data_length, bins=100)
plt.savefig('text_length_distribution.png')

from datasets import load_dataset
ds = load_dataset("json", data_files="vgen_nodupl_noiop.jsonl")
print(len(ds["train"]))