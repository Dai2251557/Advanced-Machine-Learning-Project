"""Wire contract for independent v3 assessments. No workflow, Judge or I/O dependencies."""

import json
import re
from decimal import Decimal, ROUND_HALF_UP


SCORE_KEYS = ("coverage", "factuality", "citation_support", "consistency")
AGGREGATION = "arithmetic_mean_of_four_round_half_up_2dp"


def _assessment(subscores, overall, rationale, number_types):
    if not isinstance(subscores, dict) or set(subscores) != set(SCORE_KEYS):
        raise ValueError("Judge 分项必须恰为冻结的四项")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Judge 缺少评分依据")
    cent = Decimal("0.01")
    numbers = {}
    for key, value in {**subscores, "overall": overall}.items():
        if type(value) not in number_types:  # JSON booleans are not scores.
            raise ValueError(f"Judge 的 {key} 不是数字")
        n = Decimal(str(value))
        if not (n.is_finite() and 0 <= n <= 1 and n == n.quantize(cent)):
            raise ValueError(f"Judge 的 {key} 不是有效的两位小数范围分数")
        numbers[key] = n
    mean = (sum(numbers[k] for k in SCORE_KEYS) / 4).quantize(cent, rounding=ROUND_HALF_UP)
    if numbers["overall"] != mean:
        raise ValueError("Judge overall 与四项算术均分不符")
    return {"overall_score": float(mean), "subscores": {k: float(numbers[k]) for k in SCORE_KEYS},
            "rationale": rationale, "aggregation": AGGREGATION}


def _unique_fields(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("Judge 响应包含重复字段")
    return obj


def parse_assessment(content):
    """Validate exact response fields and decimal values before converting to a JSON-ready label."""
    obj = json.loads(content, parse_float=Decimal, parse_int=Decimal, object_pairs_hook=_unique_fields)
    if not isinstance(obj, dict) or set(obj) != {*SCORE_KEYS, "overall", "rationale"}:
        raise ValueError("Judge 评分字段不完整")
    return _assessment({k: obj[k] for k in SCORE_KEYS}, obj["overall"], obj["rationale"], (Decimal,))


def validate_archived_assessment(label):
    """Check the sidecar alone; checking the actual request body requires the original receipt."""
    for field in ("final_text_sha256", "request_sha256"):
        digest = label.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Judge 缺少有效的 {field}")
    if label.get("aggregation") != AGGREGATION:
        raise ValueError("Judge 评分聚合方式不匹配")
    _assessment(label.get("subscores"), label.get("overall_score"), label.get("rationale"), (float, int))
