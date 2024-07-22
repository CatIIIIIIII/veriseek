import re
import json
import pickle
from tqdm import tqdm
from pyverilog.vparser.parser import parse


def extract_name(src_code):
    pattern = r'module\s+(\w+)'
    match = re.search(pattern, src_code)
    if match:
        return match.group(1)
    return None

gen_file_path = "data/finetune/instruct_dataset.jsonl"
with open(gen_file_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

ast_file = "instruct_dataset_ast.jsonl"
data_ast_valid = []
opencores_tree = {}
with tqdm(total=len(dataset)) as pbar:
    for data in dataset:
        pbar.update(1)
        code = data['output']
        try:
            module_name = extract_name(code)
            ast, _ = parse([code])
            data_ast_valid.append(data)
            opencores_tree[module_name] = ast
        except:
            continue

with open(ast_file, 'w', encoding='utf-8') as f:
    json.dump(data_ast_valid, f)

with open("instruct_dataset_ast.pkl", "wb") as f:
    pickle.dump(opencores_tree, f)

