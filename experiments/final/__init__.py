"""冻结终评入口。发布的 CSV/图只读，重新生成的产物放在 outputs/ 的新目录。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'results/final'
OUTPUTS = ROOT / 'outputs'
BATCH = OUTPUTS / 'course-final'
POLICIES = ('rule', 'v1', 'v2')
# SHA-256 of the original policy.json bytes recorded in the sealed execution.
CHECKPOINT_SHA256 = {
    'v1': 'ed01f52ef40841740e8bd272437b038524e50f12c7085b6b998c35bdd9564e0e',
    'v2': '70fdab5d4225972c0ae42d9f3d455fd532eee8e3484c406f8df93459d52ed6c8',
}


def new_output(path):
    """Check before reading inputs or credentials; writers also create the directory exclusively."""
    path = Path(path).resolve()
    if OUTPUTS not in path.parents:
        raise ValueError(f'生成目录必须位于 {OUTPUTS} 下：{path}')
    if path.exists():
        raise FileExistsError(f'不会重跑或覆盖已有目录：{path}')
    return path
