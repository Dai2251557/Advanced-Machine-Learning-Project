import arxiv
import json
import time
from pathlib import Path
from tqdm import tqdm

OUT_FOLDER = Path(r"./data/arxiv_corpus.json")
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def safe_str(text):
    """清洗特殊Unicode连字符，避免Windows编码异常"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u2011", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    return text.replace("\n", " ").strip()


def fetch_papers(query: str, max_results: int = 200) -> list[dict]:
    """
    使用arxiv官方库获取论文，输出项目约定schema
    :param query: arxiv搜索query
    :param max_results: 需要抓取总篇数
    :return: 标准化paper字典列表
    """
    # arxiv内置：page_size分页，delay_seconds强制间隔3s，num_retries自动重试
    client = arxiv.Client(
        page_size=100,
        delay_seconds=6.0,
        num_retries=5
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    results_iter = client.results(search)

    # tqdm展示抓取进度
    for r in tqdm(results_iter, total=max_results, desc="抓取arxiv论文元数据", unit="paper"):
        paper_id = r.get_short_id()
        abs_url = r.entry_id
        pdf_url = r.pdf_url

        # primary_category取第一个
        primary_cat = None
        if r.categories:
            primary_cat = r.categories[0]

        paper_item = {
            "paper_id": safe_str(paper_id),
            "arxiv_abs_url": safe_str(abs_url),
            "arxiv_pdf_url": safe_str(pdf_url),
            "title": safe_str(r.title),
            "abstract": safe_str(r.summary),
            "authors": [safe_str(a.name) for a in r.authors],
            "published_date": r.published.isoformat() if r.published else None,
            "updated_date": r.updated.isoformat() if r.updated else None,
            "primary_category": safe_str(primary_cat),
            "categories": [safe_str(c) for c in r.categories],
            "doi": safe_str(r.doi) if r.doi else None,
            "journal_ref": safe_str(r.journal_ref) if r.journal_ref else None
        }
        papers.append(paper_item)
    return papers


if __name__ == "__main__":
    # 修改query和max_results来调整抓取任务
    QUERY = "cat:cs.AI"
    TARGET_NUM = 600

    corpus = fetch_papers(query=QUERY, max_results=TARGET_NUM)

    with open(OUT_FOLDER, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\n抓取完成！汇总文件 {OUT_FOLDER}")
