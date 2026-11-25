"""Deterministic bilingual query expansion and BM25; no task IDs or gold data.

The general terminology dictionary is inspectable and intentionally finite.
Uncovered Chinese terms are not silently translated by an external model.
"""

import hashlib
import json
import math
import re
from collections import Counter

TERMS = {
    "强化学习": "reinforcement learning", "深度学习": "deep learning",
    "交通": "traffic", "匝道": "ramp metering", "路口": "intersection junction",
    "车流": "traffic flow", "点检测器": "detector sensor", "视频": "video",
    "迁移": "transfer adaptation", "泛化": "generalization",
    "文本生成": "text generation", "句子": "sentence", "个性化": "personalized personalization",
    "评论": "review", "标注": "label annotation", "样本": "sample",
    "推荐": "recommendation recommender", "公平": "fairness fair", "用户": "user",
    "活跃度": "activity active", "协同过滤": "collaborative filtering",
    "人口统计": "demographic", "提供者": "provider", "重排": "reranking ranking",
    "卷积": "convolutional convolution", "硬件": "hardware", "加速器": "accelerator",
    "压缩": "compression", "量化": "quantization", "缓冲": "buffer memory",
    "图像": "image", "检索": "retrieval", "特征选择": "feature selection",
    "模式分析": "pattern analysis", "回归": "regression", "核函数": "kernel",
    "神经网络": "neural network", "鲁棒": "robust robustness", "扰动": "perturbation",
    "策略": "policy", "水印": "watermark watermarking", "保密": "confidentiality",
    "隐私": "privacy", "联邦学习": "federated learning", "差分隐私": "differential privacy",
    "安全多方计算": "secure multiparty computation", "噪声": "noise noisy",
    "点云": "point cloud", "三维": "3d three dimensional", "形状": "shape",
    "运动": "motion", "几何": "geometric geometry", "问答": "question answering",
    "抽取": "extractive extraction", "注意力": "attention", "答案": "answer",
    "边界": "boundary", "上下文": "context contextual", "电网": "power grid",
    "电压": "voltage", "模仿": "imitation", "库存": "inventory", "供应链": "supply chain",
    "补货": "replenishment", "对话": "dialogue conversation", "矛盾": "contradiction",
    "话语": "utterance", "样本效率": "sample efficiency", "机器人": "robot robotic robotics",
    "一致性训练": "consistency training", "输入扰动": "augmentation perturbation",
    "跨模态": "cross modal", "算法组合": "algorithm portfolio", "流水线": "pipeline",
    "搜索": "search", "自动化": "automated automation", "配置": "configuration",
    "非配对": "unpaired", "图文": "image text", "双向生成": "bidirectional generation",
    "生成对抗": "generative adversarial", "目标检测": "object detection",
    "分类": "classification", "不平衡": "imbalance imbalanced", "时间序列": "time series",
    "图神经网络": "graph neural network", "图结构": "graph structure",
    "知识图谱": "knowledge graph", "语义": "semantic", "嵌入": "embedding",
    "无监督": "unsupervised", "半监督": "semi supervised", "自监督": "self supervised",
    "哈希": "hash hashing", "共创": "co creative co creativity", "设计": "design",
    "游戏": "game", "关卡": "level", "仿真": "simulation simulator simulated",
    "真机": "real world robot", "轨迹": "trajectory trajectories", "医学": "medical",
    "病变": "lesion disease", "解剖": "anatomical anatomy", "分割": "segmentation",
    "治理": "governance", "负责任": "responsible", "伦理": "ethics ethical",
    "因果": "causal causality", "条件独立": "conditional independence",
    "置换等变": "permutation equivariant", "变量": "variable", "档案": "archive archival",
    "转码": "transcoding", "账本": "ledger blockchain", "液滴": "droplet",
    "软体": "soft", "步态": "gait", "多样": "diversity diverse", "校准": "calibration",
    "多智能体": "multi agent", "博弈": "game", "对手": "opponent",
    "具身": "embodied", "通信": "communication", "协作": "cooperative collaboration",
    "安全": "safety security", "形式化验证": "formal verification",
    "道德": "moral", "投票": "voting voter", "记录": "record",
    "技能": "skill", "知识": "knowledge", "演化": "evolution", "调度": "scheduling",
}
STOP = set("a an the and or of to in on for with by from as is are be this that it we our their its at into using use based method methods study studies results compare comparison explain describe discuss research review".split())


def tokens(text):
    result = []
    for word in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text.lower()):
        if re.search(r"[\u4e00-\u9fff]", word):
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        elif word not in STOP:
            # Conservative plural normalization shared by query and documents.
            result.append(word[:-1] if len(word) > 4 and word.endswith('s') and not word.endswith(('ss', 'is')) else word)
    return result


def expand_query(query):
    expansions = [english for chinese, english in TERMS.items() if chinese in query]
    return sorted(set(tokens(query + ' ' + ' '.join(expansions))))


class BilingualRetriever:
    def __init__(self, papers):
        self.papers = papers
        self.counts = [Counter(tokens(p['title'] + ' ' + p['title'] + ' ' + p['abstract'])) for p in papers]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(1, len(papers)) or 1
        df = Counter(term for counts in self.counts for term in counts)
        self.idf = {term: math.log(1 + (len(papers) - n + .5) / (n + .5)) for term, n in df.items()}

    @staticmethod
    def spec():
        return {'method': 'bilingual_bm25_v1', 'k1': 1.2, 'b': .75, 'title_repetitions': 2,
                'lexicon_sha256': hashlib.sha256(json.dumps(TERMS, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}

    def rank(self, query, excluded=()):
        terms = expand_query(query)
        excluded = set(excluded)
        ranked = []
        for paper, counts, length in zip(self.papers, self.counts, self.lengths):
            if paper['id'] in excluded:
                continue
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / self.avg_length))
            ranked.append({'paper_id': paper['id'], 'score': score})
        return sorted(ranked, key=lambda row: (-row['score'], row['paper_id']))
