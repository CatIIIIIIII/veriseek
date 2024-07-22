import json

with open("data/preference/ast_rank.json", 'r', encoding='utf-8') as f:
    ast_rank = json.load(f)

print(ast_rank)