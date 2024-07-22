import os
import json
import numpy as np
from transformers import AutoTokenizer

# Define the root directory
root_dir = './golden_testbench/'

# Initialize a list to hold the JSONL data
jsonl_data = []

tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/deepseek-7B/pretrain/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )

# Traverse the directory structure
data_length = []
data_short = []
for experiment_folder in os.listdir(root_dir):
    experiment_path = os.path.join(root_dir, experiment_folder)
    
    # Check if the path is a directory
    if os.path.isdir(experiment_path):
        spec_file = None
        verilog_file = None
        
        # Find the spec and verilog files
        for file in os.listdir(experiment_path):
            if file.endswith('_spec.txt'):
                spec_file = os.path.join(experiment_path, file)
            elif file.endswith('.v'):
                verilog_file = os.path.join(experiment_path, file)
        
        # Read the content of the spec file
        if spec_file:
            with open(spec_file, 'r') as f:
                instruction = f.read().strip()
        
        # Read the content of the verilog file
        if verilog_file:
            with open(verilog_file, 'r') as f:
                output = f.read().strip()
        
        lens = len(tokenizer(instruction + output)["input_ids"])
        data_length.append(lens)
        # Append the data to the JSONL list
        jsonl_data.append({
            "name": experiment_folder,
            "instruction": instruction,
            "input": "",
            "output": output
        })
        if lens < 6000:
            data_short.append({
                "name": experiment_folder,
                "instruction": instruction,
                "input": "",
                "output": output
            })

original_jsonl_file = 'original_instruct_openscores.jsonl'
# Define the output JSONL file path
output_jsonl_file = 'instruct_openscores.jsonl'

# Save the JSONL data to a file
with open(original_jsonl_file, 'w') as f:
    json.dump(jsonl_data, f)
    
# Save the JSONL data to a file
with open(output_jsonl_file, 'w') as f:
    json.dump(data_short, f)

# plot the distribution of the length of the text
import matplotlib.pyplot as plt
plt.hist(data_length, bins=np.arange(0, 6000, 100))
plt.savefig('openscores_length_distribution.png')

from datasets import load_dataset
ds = load_dataset("json", data_files="instruct_openscores.jsonl")
ratio = len(data_short)/len(jsonl_data)*100
print(f"{ratio}% data are shorter than 6000 tokens.")
