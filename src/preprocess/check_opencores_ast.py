import re
import json
import pickle
from tqdm import tqdm
from pyverilog.vparser.parser import parse


gen_file_path = "instruct_openscores_ast.jsonl"
with open(gen_file_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

data_ast_valid = []
opencores_tree = {}
with tqdm(total=len(dataset)) as pbar:
    for data in dataset:
        pbar.update(1)
        code = data['output']
        try:
            ast, _ = parse(code)
            data_ast_valid.append(data)
            opencores_tree[data['name']] = ast
            print(ast)
        except:
            continue

with open("instruct_openscores_ast.pkl", "wb") as f:
    pickle.dump(opencores_tree, f)