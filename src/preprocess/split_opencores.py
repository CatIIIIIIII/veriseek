import os
import json

file_name = 'original_instruct_openscores.jsonl'
with open(file_name, 'r') as f:
    dataset = json.load(f)

dataset_short = []
# store the data in a dictionary in seperate files
for data in dataset:
    # create folder named data["name"]
    os.makedirs(f'opencores_testbench/{data["name"]}', exist_ok=True)
    
    # save instruction as spec file
    with open(f'opencores_testbench/{data["name"]}/{data["name"]}_spec.txt', 'w') as f:
        f.write(data["instruction"])
    
    # save output as .v file
    with open(f'opencores_testbench/{data["name"]}/{data["name"]}.v', 'w') as f:
        f.write(data["output"])
    # save instruction as .txt file
    with open(f'opencores_testbench/{data["name"]}/{data["name"]}.txt', 'w') as f:
        f.write(data["instruction"])
    