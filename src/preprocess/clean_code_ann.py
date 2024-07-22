import re
import json
import copy
from transformers import AutoTokenizer

MAX_SEQ_LEN = 5000


def extract_name(src_code):
    pattern = r'module\s+(\w+)'
    match = re.search(pattern, src_code)
    if match:
        return match.group(1)
    return None


def extract_io(src_code, io_type='input'):
    pattern = rf'\n\t*\s*{io_type}\s+([^\n]+)'
    matches = re.findall(pattern, src_code, re.DOTALL)
    names = {}
    for match in matches:
        match_bak = copy.deepcopy(match)
        match = match.strip(" ,;\n\t")
        if "//" in match:
            match = match.split("//")[0].strip(" ;,\t\n")
        if "[" in match and "]" in match:
            match = match.split("]")[-1].strip()

        if "," in match:
            name = [n.strip() for n in match.split(",")]
            for n in name:
                names[n] = match_bak
        else:
            names[match.split(" ")[-1]] = match_bak

    return names


def extract_cmt(ann_code, io_names, io_type='input'):
    ann_lines = extract_io(ann_code, io_type=io_type)

    if set(io_names) == set(list(ann_lines.keys())):
        # check "//" exists in all ann_names
        is_cmt_names = True
        for _, line in ann_lines.items():
            if "//" not in line:
                is_cmt_names = False

        if is_cmt_names:
            ann_cmt = {name: line.split('//')[1].strip()
                       for name, line in ann_lines.items()}
            if ann_cmt == {}:
                return None
            return ann_cmt

    return None


ref_path = 'is_ref_2048.json'
spec_path = 'spec_data_2048.json'
ann_path = 'ann_data_2048.json'

# Load the reference data
with open(ref_path) as f:
    ref_data = json.load(f)
with open(spec_path) as f:
    spec_data = json.load(f)
with open(ann_path) as f:
    ann_data = json.load(f)

# select from spec_data, where id is in ref_data
ref_ids = {item['id'] for item in ref_data}
src_data = {item['id']: item['src_code'] for item in ann_data}
ann_data = {item['id']: item['code_ann'] for item in ann_data}
    
# 1. filter out the spec_data where the length of spec and src_code is less than MAX_SEQ_LEN
instruct_data = [{
    "id": item['id'],
    'spec': item['spec'],
    'src_code': src_data[item['id']]
} for item in spec_data if len(item['spec'].split()) + len(src_data[item['id']].split()) < MAX_SEQ_LEN]
print(
    f"Num of samples after length filtering: {len(instruct_data)}. ({len(instruct_data)/len(spec_data):.2f})")

# 2. filter out if the function is a placeholder
instruct_data = [
    item for item in instruct_data if 'placeholder' not in item['spec']]

# 3. extract module name from src_code
for data in instruct_data:
    data['module_name'] = extract_name(data['src_code'])
noname_data = [data for data in instruct_data if data['module_name'] is None]
instruct_data = [
    data for data in instruct_data if data['module_name'] is not None]
print(
    f"Num of samples after module name filtering: {len(instruct_data)}. ({len(instruct_data)/len(spec_data):.2f})")

# 3. extract input_ports and output_ports from ann_data
for data in instruct_data:
    data['input_ports'] = list(extract_io(ann_data[data['id']]).keys())
    data['output_ports'] = list(
        extract_io(ann_data[data['id']], io_type='output').keys())

# 4. extract comments from ann_data
for data in instruct_data:
    data['input_cmts'] = extract_cmt(
        ann_data[data['id']], data['input_ports'])
    data['output_cmts'] = extract_cmt(
        ann_data[data['id']], data['output_ports'], io_type='output')
nocomment_data = [data for data in instruct_data if data['input_cmts']
                  is None or data['output_cmts'] is None]
instruct_data = [data for data in instruct_data if data['input_cmts']
                 is not None and data['output_cmts'] is not None]
print(
    f"Num of samples after comment filtering: {len(instruct_data)}. ({len(instruct_data)/len(spec_data):.2f})")
instruct_data_id = [data["id"] for data in instruct_data]

poor_data_id = [idx for idx in src_data.keys() if idx not in instruct_data_id]
poor_data = [{"id": idx,
              "src_code": src_data[idx],
              "ann_code": ann_data[idx]} for idx in poor_data_id]

with open('instruct_data.jsonl', 'w') as f:
    json.dump(instruct_data, f, indent=4)
with open('poor_instruct_data.jsonl', 'w') as f:
    json.dump(poor_data, f, indent=4)

# construct the final data
instruct_dataset = []
for data in instruct_data:

    # formualte input ports as
    inputs = "\n".join(
        [f"    {port}: {data['input_cmts'][port]}" for port in data['input_ports']])
    outputs = "\n".join(
        [f"    {port}: {data['output_cmts'][port]}" for port in data['output_ports']])

    instruct = f"""
Please act as a professional verilog designer.

Implement a module using verilog language. 

Module name:
    {data["module_name"]}

Input ports:
    {inputs}

Output ports:
    {outputs}

Implementation:
    {data["spec"]}

Give me the complete code.

"""
    instruct_dataset.append(
        {"instruction": instruct, "input": "", "output": data['src_code']})
    
# 5. extract if tokenizer(instruction + output) < 2048
tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/codellama-7B/finetune/best/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )

length = []
instruct_dataset_short = []
for instruct in instruct_dataset:
    lens = len(tokenizer(instruct["instruction"] + instruct["output"])["input_ids"])
    if lens < 2048:
        instruct_dataset_short.append(instruct)
    length.append(lens)
print(f"Num of samples after token length filtering: {len(instruct_dataset_short)}. ({len(instruct_dataset_short)/len(spec_data):.2f})")

## save as extra pre-training data
ann_pretrain = []
length = []
for idx in ref_ids:
    if idx in src_data:
        lens = len(tokenizer(src_data[idx])["input_ids"])
        length.append(lens)
        if lens < 4096:
            ann_pretrain.append({'text': src_data[idx]})
with open('ann_pretrain.jsonl', 'w') as f:
    json.dump(ann_pretrain, f, indent=4)
    
# plot the distribution of the length
import matplotlib.pyplot as plt
plt.hist(length, bins=20)
plt.show()
plt.savefig("ann_length_distribution.png")

with open('instruct_dataset.jsonl', 'w') as f:
    json.dump(instruct_dataset_short, f, indent=4)
