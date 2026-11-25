import json

# FILE_PATH = r".\data\arxiv_corpus.json"
FILE_PATH = r".\data\train_tasks.json"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    paper_list = json.load(f)

count = len(paper_list)
print(f"总记录条数：{count}")
