# evaluation/judge.py
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import hashlib
import requests

# ========= Day2冻结，冻结后禁止修改下面两行 =========
JUDGE_MODEL = "deepseek-v4-pro"
RUBRIC_PROMPT = """你是学术综述评审专家。根据以下四个维度对终稿评分（0–1），仅输出 JSON：
1. coverage：是否覆盖主题核心方面
2. factuality：陈述是否准确，不违背参考论文摘要
3. citation_support：论点是否有引用支撑；引用不存在于参考论文列表中的论文应显著扣分
4. consistency：全文逻辑是否连贯

主题={topic}
参考论文摘要列表：
{source_papers_text}
待评综述终稿：
{final_text}

评分规则：
- 四个分项均保留两位小数，范围 [0.00, 1.00]。
- overall 必须严格等于四项算术平均，即 (coverage + factuality + citation_support + consistency) / 4，四舍五入保留两位小数。不得主观调整 overall。

输出格式严格JSON：
{{"coverage":0.00,"factuality":0.00,"citation_support":0.00,"consistency":0.00,"overall":0.00}}
"""
PROMPT_HASH = hashlib.md5(RUBRIC_PROMPT.encode("utf-8")).hexdigest()


def judge_final_review(topic: str,
                       final_text: str,
                       source_papers: list[dict],
                       review_id: str,
                       api_key: str) -> dict | None:
    """
    requests直接调用deepseek，规避openai库版本bug
    :param api_key: DEEPSEEK_JUDGE_API_KEY
    """
    final_text = (final_text or "").strip()
    if not final_text:
        return None

    source_papers_text = "\n".join([
        f"paper_id:{p['paper_id']} title:{p['title']}\nabstract:{p['abstract']}"
        for p in source_papers
    ])

    prompt_filled = RUBRIC_PROMPT.format(
        topic=topic,
        source_papers_text=source_papers_text,
        final_text=final_text
    )

    url = "https://api.gpt.ge/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt_filled}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    
    data = resp.json()
    scores = json.loads(data["choices"][0]["message"]["content"])

    scores["review_id"] = review_id
    scores["judge_model"] = JUDGE_MODEL
    scores["prompt_hash"] = PROMPT_HASH
    return scores


def save_judge_result(judge_result: dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judge_result, f, ensure_ascii=False, indent=2)
