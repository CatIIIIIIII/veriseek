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


def extract_param(src_code):
    pattern = rf'\n\t*\s*parameter\s+(\w+)\s*=\s*([^\n]+)\n'
    matches = re.findall(pattern, src_code, re.DOTALL)
    params = {name: value.split("//")[0].strip(" ,") for name, value in matches}
    return params


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

PIPELINES = [
    # '0', 
    # '1',
    # '2',
    '3'
]

src_code_file_path = "pre_total_clean/total_cleaned_ann_code.txt"
src_code_file_path_output = "pre_total_clean/total_cleaned_ann_code.jsonl"

if '0' in PIPELINES:
    file_path = src_code_file_path
    # 0. load src code and split by 'endmodule'
    print(f"0. Running data loading ...")
    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            src_code = json.load(f)
    with open(file_path, "r") as f:
        src_code = f.readlines()
    # split code by 'endmodule'
    src_code = "".join(src_code).split("endmodule")
    src_code = [c+'\nendmodule' for c in src_code if len(c) > 0]
    # replace mutliple new lines with single new line
    src_code = [re.sub(r'\n+', '\n', c) for c in src_code]
    with open(src_code_file_path_output, 'w', encoding='utf-8') as f:
        json.dump(src_code, f, indent=4)
else:
    with open(src_code_file_path_output, 'r', encoding='utf-8') as f:
        src_code = json.load(f)

if '1' in PIPELINES:
    # 1. first length filter to provent compilation timeout
    print(f"1. Running length filtering ...")
    ori_lenth = len(src_code)
    src_code = [c for c in src_code if len(c) < 8192]
    print(f"Length filter: {ori_lenth} -> {len(src_code)} ({len(src_code)/ori_lenth:.2f})")


# if '2' in PIPELINES:
# 2. check compilation errors
# =============================================================================
# print(f"2. Checking compilation ...")

# MAKE_FILE = """
# .PHONY: modelsim sim clean

# TEST_DESIGN = test

# modelsim:
# 		vlib work
# 		vlog +acc -work work -sv ${TEST_DESIGN}.v

# clean:
# 		rm -rf *.log  work transcript vsim.wlf *.vcd
# """

# MAKE_COMPILE_SH = """
# #!/bin/bash

# # Resolve the absolute path of the directory where the script is located
# START_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
# EXP_DIR="${START_DIR}/$1"
# cd "$EXP_DIR"

# echo "# Start compiling"
# make modelsim
# """

# def execute_scripts_parallel(script_bash, exp_names, timeout):
#     def run_bash_script(script_path, folder_name):
#         try:
#             result = subprocess.run(['bash', script_path, folder_name], capture_output=True, text=True, check=True)
#             return (folder_name, result.stdout, result.stderr, None)
#         except subprocess.CalledProcessError as e:
#             return (folder_name, e.stdout, e.stderr, e.returncode)
#         except Exception as e:
#             return (folder_name, '', str(e), None)
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_script = {executor.submit(run_bash_script, script_bash, exp): exp for exp in exp_names}
#         try:
#             results = []
#             for future in concurrent.futures.as_completed(future_to_script, timeout=timeout):
#                 script = future_to_script[future]
#                 try:
#                     data = future.result()
#                     results.append(data)
#                 except Exception as exc:
#                     results.append((script, '', str(exc), None))
#             return results
#         except concurrent.futures.TimeoutError:
#             for future in future_to_script:
#                 future.cancel()
#             # Kill all running bash scripts
#             for future in future_to_script:
#                 script = future_to_script[future]
#                 pid = future._state
#                 if pid is not None:
#                     try:
#                         os.kill(pid, signal.SIGTERM)
#                     except OSError:
#                         pass
#             raise TimeoutError("Execution exceeded the time limit")

# # set up experiment
# exp_root = "instruct_syntax"
# # create a directory for the experiment if it does not exist, otherwise delete the existing one and create a new one
# if osp.exists(exp_root):
#     shutil.rmtree(exp_root)
# os.makedirs(exp_root)        
# with open(osp.join(exp_root, f"make_compile.sh"), 'w', encoding='utf-8') as f:
#     f.writelines(MAKE_COMPILE_SH)

# num_parallel = 16
# time_limit = 5
# syntax_pass = []
# name_src_code = {}
# with tqdm(total=len(src_code)) as pbar:
#     for i in range(0, len(src_code), num_parallel):
#         pbar.update(num_parallel)
#         # save each data in one folder

#         exp_names = []
#         for sample in src_code[i:i+num_parallel]:
#             module_name = extract_name(sample)
#             if module_name is None:
#                 continue
#             exp = f"sample_{module_name}"
#             exp_names.append(exp)
#             name_src_code[module_name] = sample
            
#             exp_path = osp.join(exp_root, exp)
#             os.makedirs(exp_path, exist_ok=True)
            
#             # save the generation
#             with open(osp.join(exp_path, f"test.v"), 'w', encoding='utf-8') as f:
#                 f.writelines(sample)
                
#             # save the make file
#             with open(osp.join(exp_path, f"makefile"), 'w', encoding='utf-8') as f:
#                 f.writelines(MAKE_FILE)
            
#         # run model sim
#         try:
#             results = execute_scripts_parallel(osp.join(exp_root, f"make_compile.sh"), exp_names, time_limit)
#             for res in results:
#                 name, ret = re.sub(r'^sample_', '', res[0]), res[1]
#                 if "Errors: 0, Warnings: 0" in ret:
#                     syntax_pass.append(name_src_code[name])
            
#         except TimeoutError as e:
#             print(e)
            
# ori_lenth = len(src_code)
# print(f"Length filter: {ori_lenth} -> {len(syntax_pass)} ({len(syntax_pass)/ori_lenth:.2f})")
# with open(src_code_file_path_output, 'w', encoding='utf-8') as f:
#     json.dump(syntax_pass, f, indent=4)
# =============================================================================

# 3. generate instruction
cleaned_src_code = [{"module_name": extract_name(c).lower(), "src_code": c} for c in src_code]

ann_path = 'pre_total_clean/ann_data_2048.json'
spec_path = 'pre_total_clean/spec_data_2048.json'
with open(ann_path) as f:
    ann_data = json.load(f)
with open(spec_path) as f:
    spec_data = json.load(f)
spec_data = {item['id']: item for item in spec_data}
name2idx= {extract_name(item["src_code"]): item["id"] for item in ann_data}
name2idx = {k.lower(): v for k, v in name2idx.items() if k is not None}
idx2anncode = {item["id"]: item["code_ann"] for item in ann_data}

for data in cleaned_src_code:
    module_name = data["module_name"]
    idx = name2idx[module_name]
    data["id"] = idx
    data["spec"] = spec_data[idx]["spec"]
    data["ann_code"] = idx2anncode[idx]

for data in cleaned_src_code:
    data['input_ports'] = list(extract_io(data['src_code']).keys())
    data['output_ports'] = list(
        extract_io(data['src_code'], io_type='output').keys())
    data['params'] = extract_param(data['src_code'])
    
    data['input_cmts'] = extract_cmt(
        data['ann_code'], data['input_ports'])
    data['output_cmts'] = extract_cmt(
        data['ann_code'], data['output_ports'], io_type='output')

# construct the final data
instruct_dataset = []
tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/deepseek-7B/pretrain/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )
for data in cleaned_src_code:

    # formualte input ports as
    inputs, outputs, params = "", "", ""
    for port in data['input_ports']:
        if data['input_cmts'] is not None:
            if port in data['input_cmts']:
                inputs += f"    {port}: {data['input_cmts'][port]}\n"
        else:
            inputs += f"    {port}: \n"
    for port in data['output_ports']:
        if data['output_cmts'] is not None:
            if port in data['output_cmts']:
                outputs += f"    {port}: {data['output_cmts'][port]}\n"
        else:
            outputs += f"    {port}: \n"
    for name, value in data['params'].items():
        params += f"    {name}: {value}\n"
        

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
    
    lens = len(tokenizer(instruct + data['src_code']))
    
    if lens < 2048:
        instruct_dataset.append({"instruction": instruct, "input": "", "output": data['src_code']})

print(f"Num of samples after comment filtering: {len(instruct_dataset)}. ({len(instruct_dataset)/len(src_code):.2f})")
with open('pre_total_clean/instruct_dataset.jsonl', 'w') as f:
    json.dump(instruct_dataset, f, indent=4)
    
instruct_pre = [{'text':item["output"]} for item in instruct_dataset]
with open('pre_total_clean/instruct_pre.jsonl', 'w') as f:
    json.dump(instruct_pre, f, indent=4)