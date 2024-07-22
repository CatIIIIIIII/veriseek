import os
import re
import json
import shutil
import signal
import subprocess
from tqdm import tqdm
import os.path as osp
import concurrent.futures
from collections import defaultdict


MAKE_FILE = """
.PHONY: modelsim sim clean

TEST_DESIGN = test

modelsim:
		vlib work
		vlog +acc -work work -sv ${TEST_DESIGN}.v

clean:
		rm -rf *.log  work transcript vsim.wlf *.vcd
"""

MAKE_COMPILE_SH = """
#!/bin/bash

# Resolve the absolute path of the directory where the script is located
START_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
EXP_DIR="${START_DIR}/$1"
cd "$EXP_DIR"

echo "# Start compiling"
make modelsim
"""

def execute_scripts_parallel(script_bash, exp_names, timeout):
    def run_bash_script(script_path, folder_name):
        try:
            result = subprocess.run(['bash', script_path, folder_name], capture_output=True, text=True, check=True)
            return (folder_name, result.stdout, result.stderr, None)
        except subprocess.CalledProcessError as e:
            return (folder_name, e.stdout, e.stderr, e.returncode)
        except Exception as e:
            return (folder_name, '', str(e), None)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_script = {executor.submit(run_bash_script, script_bash, exp): exp for exp in exp_names}
        try:
            results = []
            for future in concurrent.futures.as_completed(future_to_script, timeout=timeout):
                script = future_to_script[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    results.append((script, '', str(exc), None))
            return results
        except concurrent.futures.TimeoutError:
            for future in future_to_script:
                future.cancel()
            # Kill all running bash scripts
            for future in future_to_script:
                script = future_to_script[future]
                pid = future._state
                if pid is not None:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
            raise TimeoutError("Execution exceeded the time limit")


def collect_results(results, flattened_data):
    success = defaultdict(list)
    fail = defaultdict(list)

    for folder_name, stdout, _, _ in results:
        
        folder_name = folder_name.split('_')
        uni_idx, idx = int(folder_name[-2]), int(folder_name[-1])
        if "Errors: 0, Warnings: 0" in stdout:
            success[idx].append(flattened_data[uni_idx])
        else:
            fail[idx].append(flattened_data[uni_idx])
    
    pairs = []
    for idx in success:
        if idx in fail:
            pairs.append((success[idx], fail[idx]))
    pairs = [p for p in pairs if p is not None]
    
    return pairs

def extract_name(src_code):
    pattern = r'module\s+(\w+)'
    match = re.search(pattern, src_code)
    if match:
        return match.group(1)
    return None

# gen_file_path = "instruct_dataset.jsonl"
gen_file_path = "instruct_openscores.jsonl"
with open(gen_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# set up experiment
exp_root = "instruct_syntax"
# create a directory for the experiment if it does not exist, otherwise delete the existing one and create a new one
if osp.exists(exp_root):
    shutil.rmtree(exp_root)
os.makedirs(exp_root)        
with open(osp.join(exp_root, f"make_compile.sh"), 'w', encoding='utf-8') as f:
    f.writelines(MAKE_COMPILE_SH)

num_parallel = 16
time_limit = 5
name_instruction = {}
name_output = {}
syntax_pass = []
with tqdm(total=len(data)) as pbar:
    for i in range(0, len(data), num_parallel):
        pbar.update(num_parallel)
        # save each data in one folder

        exp_names = []
        for sample in data[i:i+num_parallel]:
            module_name = extract_name(sample['output'])
            if module_name is None:
                continue
            exp = f"sample_{module_name}"
            exp_names.append(exp)
            name_instruction[module_name] = sample['instruction']
            name_output[module_name] = sample['output']
            
            exp_path = osp.join(exp_root, exp)
            os.makedirs(exp_path, exist_ok=True)
            
            # save the generation
            with open(osp.join(exp_path, f"test.v"), 'w', encoding='utf-8') as f:
                f.writelines(sample["output"])
                
            # save the make file
            with open(osp.join(exp_path, f"makefile"), 'w', encoding='utf-8') as f:
                f.writelines(MAKE_FILE)
            
        # run model sim
        try:
            results = execute_scripts_parallel(osp.join(exp_root, f"make_compile.sh"), exp_names, time_limit)
            for res in results:
                name, ret = re.sub(r'^sample_', '', res[0]), res[1]
                if "Errors: 0, Warnings: 0" in ret:
                    syntax_pass.append({"instruction": name_instruction[name],
                                        "input": "",
                                        "output": name_output[name]})
            
        except TimeoutError as e:
            print(e)


print(f"Syntax pass: {len(syntax_pass)}")
with open("instruct_openscores.jsonl", 'w', encoding='utf-8') as f:
    json.dump(syntax_pass, f, indent=4)