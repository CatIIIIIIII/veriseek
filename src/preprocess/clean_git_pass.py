import json
import matplotlib.pyplot as plt

with open("/data/RTL_code/pass_iverilog.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)
for data in dataset:
    data['text'] = data['src_code'].replace('    ', '\t')
    data.pop('src_code')

# length_dataset = [len(data['src_code'].split(' ')) for data in dataset]
dataset_uniform = [data for data in dataset if len(data['text'].split(' ')) <= 3000]
dataset_long = [data for data in dataset if len(data['text'].split(' ')) > 3000]

# chunk every dataset_long to 3000 words
# for data in dataset_long:
#     src_code = data['text'].split(' ')
#     module_name = data['module_name']
#     for i in range(0, len(src_code), 3000):
#         dataset_uniform.append({
#             'text': ''.join(src_code[i:i+3000]), })

length_dataset = [len(data['text'].split(' ')) for data in dataset]
plt.figure()
plt.hist(length_dataset, bins=100)
plt.xlabel('Length of source code')
plt.ylabel('Number of samples')
plt.title('Length of source code distribution')
plt.savefig('length_dist_git_pass.png')

# save the dataset with long source code
with open("git_pass_cleaned.json", 'w', encoding='utf-8') as f:
    json.dump(dataset_uniform, f, indent=4)
    
print(len(dataset_long)/len(dataset_uniform))
print(len(dataset_uniform))
# print(dataset_long[0]['src_code'].split(' '))