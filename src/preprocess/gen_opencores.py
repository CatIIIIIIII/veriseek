import re
import json
from transformers import AutoTokenizer

def extract_name(instruct, output):

    pattern = r"`([^`]+)`"
    match = re.search(pattern, instruct)
    assert match, "No module name found."
    module_name = match.group(1)
    assert module_name in output, f"Module name {module_name} not found in output."
    
    return module_name

def extract_desc(desc):   
    assert "2. Module Functions" in desc, "No module functions found."
    desc = desc.replace("2. Module Functions", "").strip()
    # split at the first occurrence of "module"
    desc = desc.split("module", 1)[1].strip().replace("is designed to", "to").replace("is to", "to").replace("**", "").replace("- ", "")
    desc = "Implement a module " + desc
    
    return desc

def extract_ios(name, ports):
    def extract_port(port_str):
        port_out = []
        for p in port_str.split("\n"):
            if '-' in p and "Inputs" not in p and "Outputs" not in p and "Parameters" not in p and "Output Port" not in p and "Input Port" not in p:
                p = p.strip()
                p = p.replace("**", "").replace("*", "").replace("`", "").replace("-", "").strip()
                port_out.append(p)
        return port_out
    
    ports = ports.replace("3. Input and Output Port Descriptions", "").strip()
    ports_list = ports.split("\n- **")
    if len(ports_list) == 2:
        if "Input" in ports_list[0]:
            input_ports = extract_port(ports_list[0])
            output_ports = extract_port(ports_list[1])
            parameter_ports = []
        else:
            input_ports = extract_port(ports_list[1])
            output_ports = extract_port(ports_list[0])
            parameter_ports = []
    elif len(ports_list) == 3:
        input_ports = extract_port(ports_list[0])
        output_ports = extract_port(ports_list[1])
        parameter_ports = extract_port(ports_list[2])
    else:
        print(name)
        raise ValueError("Invalid ports format.")
    
    ret = """Input ports:\n"""
    if len(input_ports) > 0:
        if input_ports[0] != "None":
            for input in input_ports:
                ret += f"\t{input}\n"
    if len(output_ports) > 0:
        if output_ports[0] != "None":
            ret += """Output ports:\n"""
            for output in output_ports:
                ret += f"\t{output}\n"
    if parameter_ports:
        ret += """Parameter ports:\n"""
        for parameter in parameter_ports:
            ret += f"\t{parameter}\n"

    return ret
    
def extract_implement(impl):
    impl = impl.replace("4. Internal Working Principle", "").replace("5. Implementation Logic Explanation", "").strip()  
    impl = impl.replace("**", "").replace("*", "").replace("`", "").replace("\n\n", "\n").replace("- ", "").strip()

    return impl

file_name = 'instruct_openscores.jsonl'
with open(file_name, 'r') as f:
    dataset = json.load(f)

sum = 0
length = []
instruct_openscores = []
for data in dataset:
    output = data['output']
    instruction = data['instruction'].split("###")
    instruction = [i.strip() for i in instruction]
    instruction = [i for i in instruction if i != '']
    
    # 1. module name
    name_instruction = instruction[0]
    name = extract_name(name_instruction, output)
    
    # 2. module description
    desc_instruction = instruction[1]
    desc = extract_desc(desc_instruction)
    
    # 3. module ports
    ports_instruction = instruction[2]
    ports = extract_ios(name, ports_instruction)
    # print()
    # if "4. Internal Working Principle" in instruction[3]:
    #     principle = extract_implement(instruction[3])
        
    if "5. Implementation Logic Explanation" in instruction[4]:
        impl = extract_implement(instruction[4])
    
    # break
    instruct_new = f"""Please act as a professional verilog designer.
    
{desc}

Module name:  
\t{name}

{ports}

Implementation:
{impl}

Give me the complete code.

"""
    instruct_openscores.append({"name": name.lower(),
                                "instruction":instruct_new, 
                                "input":"", "output":output})
    
print(f"Total: {sum/len(dataset)}")

# 5. extract if tokenizer(instruction + output) < 2048
tokenizer = AutoTokenizer.from_pretrained(
            "/data/ckpt/wangning/codellama-7B/finetune/best/",
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
        )
length = []
instruct_dataset_short = []
for instruct in instruct_openscores:
    lens_prompt = len(tokenizer(instruct["instruction"])["input_ids"])
    lens_output = len(tokenizer(instruct["output"])["input_ids"])
    if lens_prompt < 2048 and lens_output < 2048:
        instruct_dataset_short.append(instruct)
print(f"Num of samples after token length filtering: {len(instruct_dataset_short)}. ({len(instruct_dataset_short)/len(dataset):.2f})")

with open('instruct_openscores.jsonl', 'w') as f:
    json.dump(instruct_dataset_short, f)