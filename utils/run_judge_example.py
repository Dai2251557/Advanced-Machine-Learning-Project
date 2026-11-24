# run_judge_example.py
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from evaluation.judge import judge_final_review, save_judge_result

# 加载.env
load_dotenv()
judge_api_key = os.getenv("DEEPSEEK_JUDGE_API_KEY")
if not judge_api_key:
    raise SystemExit("错误：未读取到 DEEPSEEK_JUDGE_API_KEY，请检查 .env 文件")

# 这里是测试文本
review_id = "R001"
topic = "写一篇DQN在多智能体系统应用的文献综述"

final_text = """
# DQN多智能体综述
深度Q网络(DQN)在多智能体强化学习中得到大量应用。
很多工作针对环境非平稳问题做改进，包括集中式训练分布式执行框架。
"""

source_papers = [
    {
        "paper_id": "2012.13026v1",
        "title": "Rethink AI‑based Power Grid Control",
        "abstract": "Recently, deep reinforcement learning (DRL)‑based approach has shown promise solving complex decision ..."
    }
]

jr = judge_final_review(
    topic=topic,
    final_text=final_text,
    source_papers=source_papers,
    review_id=review_id,
    api_key=judge_api_key
)

if jr is not None:
    out_dir = Path("./outputs/judge_labels/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{review_id}_judge.json"
    save_judge_result(jr, str(out_file))
    print("===Judge打分结果===")
    print(json.dumps(jr, indent=2, ensure_ascii=False))
else:
    print("无有效终稿，跳过打分")
