import re
import json
import shutil
import signal
import pickle
import subprocess
from tqdm import tqdm
import os.path as osp
import concurrent.futures
from collections import defaultdict

import pyverilog.vparser.ast as vast
from pyverilog.vparser.parser import parse

class ParseTreeNormalizer:
    def normalize(self, node):
        if isinstance(node, vast.Node):
            normalized_children = [self.normalize(c) for c in node.children()]
            if isinstance(node, vast.Identifier):
                return ('Identifier',)  # Ignore the actual name
            if isinstance(node, vast.IntConst):
                return ('IntConst',)  # Ignore the actual value
            return (node.__class__.__name__, tuple(normalized_children))
        elif isinstance(node, list):
            return [self.normalize(n) for n in node]
        return node

def compare_trees(tree1, tree2):
    if tree1 == tree2:
        return 1.0  # Trees are identical
    if isinstance(tree1, tuple) and isinstance(tree2, tuple):
        if tree1[0] == tree2[0]:
            children1 = tree1[1]
            children2 = tree2[1]
            if len(children1) == len(children2):
                return sum(compare_trees(c1, c2) for c1, c2 in zip(children1, children2)) / len(children1)
            else:
                # Penalize for different number of children
                return sum(compare_trees(c1, c2) for c1, c2 in zip(children1, children2)) / max(len(children1), len(children2))
    return 0.0  # Trees are different

file_path = "instruct_generations.json"
label_path = "instruct_dataset_ast.pkl"
with open(file_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
with open(label_path, "rb") as f:
    label_ast = pickle.load(f)
label_ast = list(label_ast.values())

# name2data
dataset_pairs = []
with tqdm(total=len(dataset)) as pbar:
    for data, label in zip(dataset, label_ast):
        pbar.update(1)
        instance = {}
        instance["instruction"] = data[0]["prompt"]
        normalizer = ParseTreeNormalizer()
        normalized_label = normalizer.normalize(label)
        
        # select generations which are difference
        gens = [d["generation"] for d in data]
          
        gens_info = []
        for gen in gens:
            try:
                gen_ast, _ = parse([gen])
                normalized_gen = normalizer.normalize(gen_ast)
                reward = compare_trees(normalized_gen, normalized_label)
                gens_info.append({
                    "generation": gen,
                    "reward": reward
                })
            except:
                break
        instance["gen"] = gens_info
        dataset_pairs.append(instance)

with open("data/preference/ast_rank.json", 'w') as f:
    json.dump(dataset_pairs, f)