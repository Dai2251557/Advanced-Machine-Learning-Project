"""从仓库根目录执行 python -B -m utils.run_judge_example；只在 stdout 展示合成回执。"""

import json

from evaluation.judge import FIXTURE_MODEL, build_judge_request, finalize_judge_result, prepare_fixture_review


def main():
    prepared = prepare_fixture_review(
        survey_instruction="说明玩具方法的输入与输出。", final_text="输入为色块，输出为其颜色。[toy-A]\n",
        frozen_papers=[{"id": "toy-A", "title": "虚构的颜色识别", "abstract": "该玩具方法返回输入色块的颜色。"}],
        review_id="fixture-example", trajectory_id="fixture-example-01", split="train",
    )
    # 常量只演示评分 JSON 的协议与均分，并非对文本的实际评价。
    content = json.dumps({"coverage": 0.8, "factuality": 0.7, "citation_support": 0.9, "consistency": 0.6,
                          "overall": 0.75, "rationale": "离线协议示例：四个分项均为自造数值。"}, ensure_ascii=False)
    receipt = {"binding": dict(prepared["binding"]), "request": build_judge_request(prepared),
               "response": {"model": FIXTURE_MODEL, "choices": [
        {"finish_reason": "stop", "message": {"content": content}}]}}
    print(json.dumps(finalize_judge_result(prepared, receipt), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
