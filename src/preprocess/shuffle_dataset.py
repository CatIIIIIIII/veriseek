import json
import random
import jsonlines

        
def load_jsonl(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def main():
    # 文件列表
    files = ["data/pretrain/git_pass_cleaned.json", "data/pretrain/cpp.jsonl"]
    # files = ["data/finetune/code_alpaca_20k.json", "data/finetune/instruct_dataset.jsonl"]

    all_data = []
    for file in files:
        all_data.extend(load_jsonl(file))

    random.shuffle(all_data)

    # save all data as a new json file
    with open('shuffled_pretrain.jsonl', 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    from datasets import load_dataset
    ds = load_dataset("json", data_files="shuffled_pretrain.jsonl")
    print(len(ds["train"]))
    print(ds["train"][0])

if __name__ == "__main__":
    main()
