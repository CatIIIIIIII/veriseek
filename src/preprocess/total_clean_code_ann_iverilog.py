import re
import os
import json
import copy
import subprocess
from tqdm import tqdm
import os.path as osp
import concurrent.futures
from collections import defaultdict
from transformers import AutoTokenizer
from pprint import pprint

        
src_code_file_path = "pre_total_clean/total_cleaned_ann_code_iverilog.jsonl"

# load jsonlines file
src_code = []
with open(src_code_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        src_code.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/deepseek-7B/pretrain/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )

i = 0
instruct_dataset = []
for data in src_code:
    
    desc = data['description']
    desc = desc.split("Logical Function Description:")
    if len(desc) == 2 and "Interface Description:\n```\n" in desc[0]:
        interface, logic = desc
        interface = interface.replace("Interface Description:\n```\n", "").replace("```", "").strip("\n")
        logic = logic.replace("```", "").strip("\n")
        # interface = 
        instruct = f"""
Please act as a professional verilog designer.

Implement a module using verilog language. 

{interface}

Implementation:

{logic}

Give me the complete code.
""" 
        
        # replace multiple \n with single \n
        instruct = re.sub(r'\n+', '\n', instruct)
        output = re.sub(r'(\n\s{2,})+', r'\n  ', data["correct_code"])
        len_instruct = len(tokenizer(instruct))
        lens = len(tokenizer(instruct + output)["input_ids"])
        if lens < 1024 and len_instruct < 512 and "As a Verilog design expert," not in instruct and "Logical Function Description" not in instruct:
            instruct_dataset.append({"instruction": instruct, "input": "", "output": output})
            i += 1

with open('pre_total_clean/instruct_dataset.jsonl', 'w') as f:
    json.dump(instruct_dataset, f, indent=4)
        
print(f"Total {i/len(src_code)} instructions are generated.")
pprint(instruct_dataset[0])    