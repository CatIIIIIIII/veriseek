import os
import json
import shutil
import os.path as osp

file_root = "/data/raw/opencores"
file_name = 'instruct_openscores.jsonl'
output_root = "opencores_testbench"
with open(file_name, 'r') as f:
    dataset = json.load(f)
name_dataset = [d["name"] for d in dataset]
name_dataset_lower = [d["name"].lower() for d in dataset]
# get the name of all the folders in the root directory
folders = os.listdir(file_root)
folders_lower = {f.lower(): f for f in folders}

for name in name_dataset_lower:
    if name in list(folders_lower.keys()):
        MAKE_FILE = \
f""".PHONY: modelsim sim clean

modelsim:
\t\tvlib work
\t\tvlog +acc -work work -sv {name}.v testbench.v

sim:
\t\tvsim -c -do "run -all; quit" testbench

clean:
\t\trm -rf *.log  work transcript vsim.wlf *.vcd

"""     
        name = folders_lower[name]
        exp_root = osp.join(output_root, name)
        os.makedirs(exp_root, exist_ok=True)
        testbench_file = osp.join(file_root, name, "opencores_auto/testbench.sv")
        # copy the testbench file to the expected root and change the name to testbench.v
        shutil.copy(testbench_file, osp.join(exp_root, "testbench.v"))
        # save the makefile
        with open(osp.join(exp_root, "makefile"), 'w') as f:
            f.write(MAKE_FILE)
        # copy the .v file to the expected root
        shutil.copy(osp.join(file_root, name, f"{name}.v"), exp_root)
        
        print(testbench_file)