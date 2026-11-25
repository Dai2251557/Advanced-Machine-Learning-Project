import json

IN_PATH = r".\data\arxiv_corpus.json"
OUT_DIR = r".\data"

with open(IN_PATH, "r", encoding="utf-8") as f:
    all_papers = json.load(f)

total = len(all_papers)
print(f"原始总条数：{total}")

chunk_size = 100
# 切片，每100条一组
for i in range(0, total, chunk_size):
    chunk = all_papers[i:i + chunk_size]
    # 文件名：arxiv_corpus_0_100.json，100_200.json ...
    out_file = f"{OUT_DIR}\\arxiv_corpus_{i}_{i+chunk_size}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    print(f"保存 {i} ~ {i + chunk_size - 1}，共 {len(chunk)} 条 -> {out_file}")

print("全部切分完成！")
