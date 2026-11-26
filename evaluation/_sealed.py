"""只读 frozen.execute_pairs 已经完整发布的批次，不运行或重放 Runner。"""

import hashlib
import json
import re


def sealed_episodes(directory, review_ids, delivery):
    """检查交付绑定、配对、原始字节哈希和轨迹身份；调用者须整批预检后才交出终稿。"""
    complete = json.loads((directory / "execution_complete.json").read_bytes())
    if (complete["mode"] != "formal" or complete["all_episodes_complete"] is not True or
            complete["test_delivery"] != delivery):
        raise ValueError("测试尚未全部封存或不是同一正式交付")
    pairs = set()
    tids = set()
    paths = set()
    for row in complete["episodes"]:
        rid, policy, tid, name = (row[k] for k in ("review_id", "policy", "trajectory_id", "trace"))
        if (rid not in review_ids or policy not in ("rule", "dqn") or (rid, policy) in pairs or
                not isinstance(tid, str) or not tid.strip() or tid in tids or name in paths or
                not re.fullmatch(r"[0-9]{3,}-(rule|dqn)\.json", name) or not name.endswith(f"-{policy}.json")):
            raise ValueError("封存索引中的配对、轨迹或文件名无效")
        pairs.add((rid, policy))
        tids.add(tid)
        paths.add(name)
    if pairs != {(rid, policy) for rid in review_ids for policy in ("rule", "dqn")}:
        raise ValueError("缺少封存任务的配对轨迹")
    for row in complete["episodes"]:
        raw = (directory / row["trace"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("封存轨迹字节哈希不一致")
        ep = json.loads(raw)
        if (ep["review_id"] != row["review_id"] or ep["trajectory_id"] != row["trajectory_id"] or
                type(row["success"]) is not bool or ep["success"] is not row["success"]):
            raise ValueError("封存索引与轨迹身份或完成状态不符")
        steps = ep.get("transitions")
        if not steps:
            raise ValueError("封存轨迹没有终态步骤")
        last = steps[-1]
        after = last["state_after"]
        reason = ep["terminal_reason"]
        if (last["done"] is not True or after["terminal"] is not True or
                last["terminal_reason"] != reason or after["terminal_reason"] != reason):
            raise ValueError("轨迹末步没有达到所声明的终态")
        if ep["success"]:
            if (reason != "final_written" or last["action_id"] not in ("write_fast", "write_deep") or
                    last["success"] is not True):
                raise ValueError("封存终稿必须来自成功的末步写作")
        elif reason not in ("max_steps", "token_budget", "usage_unknown"):
            raise ValueError("封存的失败轨迹缺少合法的终止原因")
        yield ep
