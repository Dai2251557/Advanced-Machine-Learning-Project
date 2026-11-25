"""在训练侧补齐下一步的物理 Mask，不改变 Runner 的原始轨迹。"""


def step_with_masks(runner, action_id):
    transition = runner.step(action_id)
    return {**transition, "next_mask": runner.valid_actions()}
