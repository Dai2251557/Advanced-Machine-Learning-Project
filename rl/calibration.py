"""Empirical simulation from validated *training* Runner episodes (CONTRACT §5).

    model = fit_calibration(episodes, labels, mode="fixture", corpus_size=12)
    env = model.make_env()  # same reset/step and physical masks as SyntheticEnv

Fixture mode tests the pipeline; it is not evidence of real calibration. No
backend, Judge, artifact write or network operation is performed here.
"""

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean, median

from agents.retrieval import BilingualRetriever
from core.rule_scheduler import RuleScheduler
from core.runner import total
from core.schema import ACTIONS, SCORE_KEYS, RunnerConfig, verifier_value
from experiments.collect import json_sha256, validate_collection
from rl.synthetic_env import Outcome, SyntheticEnv


TOKEN_FIELDS = ("input_tokens", "output_tokens", "verifier_tokens")
LEVELS = ("joint", "pool_difficulties", "pool_score_bands", "same_size")
MIN_REVIEWS = 2


def _stage(state):
    if not state["retrieved_paper_count"]:
        return "empty"
    return "summary" if state["summary_available"] else "papers"


def _scores(state):
    v = state["verifier"]
    if v["checked_material_revision"] != state["material_revision"]:
        return {}
    return {k: v["scores"][k] for k in SCORE_KEYS if v["valid_scores"][k]}


def _coverage_band(scores):
    c = scores.get('coverage')
    return 'unknown' if c is None else 'low' if c < .40 else 'mid' if c < .70 else 'high'


def _keys(state, action, version):
    scores = _scores(state)
    stage = _stage(state)
    if version == 'v2' and stage == 'papers':
        band = _coverage_band(scores)
    else:
        value = fmean(scores.values()) if scores else None
        band = ("unknown" if value is None else "low" if value < 0.5 else
                "mid" if value < 0.75 else "high")
    base = (action, stage)
    sized = base + (state['retrieved_paper_count'],)
    scope = tuple(scores)
    return (sized + (scope, band, state["difficulty"]), sized + (scope, band),
            sized + (scope,), sized, base)


def _clip(value):
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class _Row:
    review_id: str
    papers_before: int
    step_success: bool  # 实测整步结果，包括校验；不同于下方动作本身是否已执行
    tokens: tuple
    action_success: bool  # execution, independently of subsequent verification
    before: dict
    after: dict | None  # None = no new check, not 'reuse the old scores'
    calls: tuple[dict, ...]  # ordered numeric/role metadata, never donor prompts or outputs

    @classmethod
    def read(cls, tr):
        before, after = tr["state_before"], tr["state_after"]
        tokens = tuple(tr[k] for k in TOKEN_FIELDS)
        if any(x is not None and (type(x) is not int or x < 0) for x in tokens):
            raise ValueError("Invalid step token count")
        step_tokens = total(tokens)
        if step_tokens != tr["total_workflow_tokens"]:
            raise ValueError("Step workflow token ledger does not reconcile")
        used = before["workflow_tokens_used"]
        if after["workflow_tokens_used"] != total((used, step_tokens)):
            raise ValueError("Cumulative workflow token ledger does not reconcile")
        if tr["action_id"].startswith("write_"):
            # The shared Runner may reject a generated draft on cost grounds;
            # simulated budget eligibility is recomputed from the simulated path.
            executed = tr["success"] or any(c["role"] == "write" and c["status"] == "ok"
                                            for c in tr["calls"])
            checked = None
        else:
            executed = after["material_revision"] > before["material_revision"]
            checked = _scores(after) if executed and tr["success"] else None
        calls = []
        for call in tr["calls"]:
            event = {k: call[k] for k in ("role", "attempt", "status", "prompt_tokens",
                                          "completion_tokens", "cache_hit")}
            elapsed = call.get("elapsed_seconds")
            event["elapsed_seconds"] = (elapsed if type(elapsed) in (int, float)
                                        and math.isfinite(elapsed) and elapsed >= 0 else None)
            calls.append(event)
        return cls(tr["review_id"], before['retrieved_paper_count'], tr['success'], tokens,
                   executed, _scores(before), checked, tuple(calls))

    def call_prefix(self, state):
        """Replay observed hits, not request contents; stop at the first blocked miss."""
        used, budget = state["workflow_tokens_used"], state["workflow_tokens_budget"]
        calls = []
        for call in self.calls:
            if not call["cache_hit"] and (used is None or used >= budget):
                break
            calls.append(call.copy())
            used = total((used, call["prompt_tokens"], call["completion_tokens"]))
        return tuple(calls), used

    def transferred_scores(self, state):
        if self.after is None:
            return None
        current = _scores(state)
        # Use a measured change only on dimensions valid on both sides AND
        # currently valid. For newly measured dimensions bootstrap the donor's
        # absolute post value. Never fill invalid dimensions from another row.
        return {k: _clip(current[k] + (v - self.before[k]))
                if k in current and k in self.before else v for k, v in self.after.items()}


class _Quality:
    """Bounded empirical offset to the current Verifier; no label in any state."""

    def __init__(self, episodes, labels, source):
        self.source = source
        self.offsets = defaultdict(list)
        self.targets = []
        by_review = defaultdict(list)
        no_verifier = defaultdict(lambda: defaultdict(list))
        for ep in episodes:
            label = labels.get(ep["trajectory_id"])
            if label is None:
                continue
            score = label["overall_score"]
            self.targets.append(score)
            tr = ep["transitions"][-1]
            state, action = tr["state_before"], tr["action_id"]
            value = verifier_value(state)
            by_review[ep['review_id']].append(score)
            if value is None:
                no_verifier[(action, _stage(state))][ep['review_id']].append(score)
            else:
                group = (action, _stage(state), tuple(_scores(state)))
                self.offsets[group].append((ep["review_id"], score - value))
        # 先在题内取均值，避免重复跑得最多的综述主导代理分。
        self.unscored_means = {key: fmean(fmean(scores) for scores in reviews.values())
                               for key, reviews in no_verifier.items() if len(reviews) >= MIN_REVIEWS}
        self.pooled_mean = fmean(fmean(scores) for scores in by_review.values()) if by_review else None

    def estimate(self, state, action):
        if not action.startswith("write_"):
            raise ValueError("Final quality is only defined for writing")
        value = verifier_value(state)
        if value is None:
            key = (action, _stage(state))
            if key in self.unscored_means:
                return self.unscored_means[key], self.source + '_no_verifier_group_mean'
            if self.pooled_mean is not None:
                return self.pooled_mean, self.source + '_pooled_mean_low_support'
            return None, 'unavailable_no_fresh_scores_or_training_labels'
        group = (action, _stage(state), tuple(_scores(state)))
        residuals = self.offsets.get(group, ())
        if len({r for r, _ in residuals}) >= MIN_REVIEWS:
            return _clip(value + fmean(d for _, d in residuals)), self.source + "_verifier_offset"
        return value, "fresh_valid_verifier_mean"


class _CoverageQuality:
    """One supervised scale for raw and summarized material; never use other dimensions."""

    ridge = .0025  # added to review-balanced Var(C) on the [0, 1] coverage scale
    prior_reviews = 4  # zero-residual pseudo-reviews per (stage, writer) cell

    def __init__(self, episodes, labels, source):
        self.source = source
        by_review, fresh = defaultdict(list), defaultdict(list)
        for ep in episodes:
            label = labels.get(ep['trajectory_id'])
            if label is None:
                continue
            rid, y = ep['review_id'], label['overall_score']
            tr = ep['transitions'][-1]
            state = tr['state_before']
            by_review[rid].append(y)
            c = _scores(state).get('coverage')
            if c is not None:
                fresh[rid].append((c, y, (_stage(state), tr['action_id'])))
        self.targets = [y for ys in by_review.values() for y in ys]
        self.pooled_mean = fmean(fmean(ys) for ys in by_review.values()) if by_review else None
        self.label_reviews, self.fresh_reviews = len(by_review), len(fresh)
        self.fresh_labels = sum(map(len, fresh.values()))
        self.intercept = self.slope = None
        self.groups = {}
        if not fresh:
            return
        # Each review contributes total weight 1 regardless of repeated trajectories.
        c_bar = fmean(fmean(c for c, _, _ in rows) for rows in fresh.values())
        y_bar = fmean(fmean(y for _, y, _ in rows) for rows in fresh.values())
        cov = fmean(fmean((c - c_bar) * (y - y_bar) for c, y, _ in rows) for rows in fresh.values())
        var = fmean(fmean((c - c_bar) ** 2 for c, _, _ in rows) for rows in fresh.values())
        self.slope = cov / (var + self.ridge)
        self.intercept = y_bar - self.slope * c_bar
        residuals = defaultdict(lambda: defaultdict(list))
        for rid, rows in fresh.items():
            for c, y, group in rows:
                residuals[group][rid].append(y - (self.intercept + self.slope * c))
        for group, reviews in residuals.items():
            n = len(reviews)
            offset = (fmean(fmean(ds) for ds in reviews.values()) * n / (n + self.prior_reviews)
                      if n >= MIN_REVIEWS else 0.0)
            self.groups[group] = (n, offset)

    def estimate(self, state, action):
        if not action.startswith('write_'):
            raise ValueError('Final quality is only defined for writing')
        if self.pooled_mean is None:
            return None, 'unavailable_no_training_labels'
        method = self.source + '_estimated_'
        c = _scores(state).get('coverage')
        if c is None or self.slope is None:
            return self.pooled_mean, method + 'review_mean_no_fresh_coverage_fit_or_input'
        n, offset = self.groups.get((_stage(state), action), (0, 0.0))
        return (_clip(self.intercept + self.slope * c + offset),
                method + 'coverage_ridge' + ('_group_residual' if n >= MIN_REVIEWS else ''))

    def manifest(self):
        return {'method': 'estimated_review_balanced_coverage_ridge', 'label_source': self.source,
                'label_reviews': self.label_reviews, 'labels': len(self.targets), 'review_mean': self.pooled_mean,
                'fresh_reviews': self.fresh_reviews, 'fresh_labels': self.fresh_labels,
                'intercept': self.intercept, 'slope': self.slope, 'ridge': self.ridge,
                'residual_prior_reviews': self.prior_reviews, 'residual_min_reviews': MIN_REVIEWS,
                'group_residuals': [{'stage': s, 'writer': a, 'reviews': n, 'offset': d}
                                    for (s, a), (n, d) in sorted(self.groups.items())]}


class Calibration:
    """Fitted outcome callable. Instantiate via fit_calibration (T4 provenance gate)."""

    def __init__(self, episodes, labels, mode, corpus_size, retrieval_spec_sha256, calibration_version='v1'):
        if calibration_version not in ('v1', 'v2'):
            raise ValueError(f'Unknown calibration_version: {calibration_version}')
        self.calibration_version = calibration_version
        self.mode = mode
        self.retrieval_spec_sha256 = retrieval_spec_sha256
        self.source = "synthetic_fixture" if mode == "fixture" else "formal_train_runner"
        self.config = RunnerConfig(**episodes[0]["config"])
        self.corpus_size = corpus_size
        self.buckets = defaultdict(list)
        self.difficulties = {ep['review_id']: ep['public_task']['difficulty'] for ep in episodes}
        self.reviews = sorted(self.difficulties)
        self.episodes = len(episodes)
        for ep in episodes:
            if ep["config"] != asdict(self.config):
                raise ValueError("Cannot pool different Runner configs")
            for tr in ep["transitions"]:
                state, action = tr["state_before"], tr["action_id"]
                row = _Row.read(tr)
                for key in _keys(state, action, calibration_version):
                    self.buckets[key].append(row)
        label_source = "synthetic_fixture" if mode == "fixture" else "train_judge"
        quality_cls = _CoverageQuality if calibration_version == 'v2' else _Quality
        self.quality = quality_cls(episodes, labels, label_source)
        self.unscorable_writes = self._unscorable_writes()

    def _unscorable_writes(self):
        if self.quality.pooled_mean is not None:
            return []
        if self.calibration_version == 'v2':
            # Unlabelled v2 is cost-only even when all four Verifier scores are fresh.
            return sorted(f'{k[1]}/{k[0]}' for k in self.buckets if len(k) == 2 and k[0].startswith('write_'))
        # 按与 _select 相同的最粗可用阶段做保守预检，涵盖跨轨迹重组出的路径。
        stale_cost, write_cost = {}, {}
        for key, rows in self.buckets.items():
            if len(key) != 2:
                continue
            action, stage = key
            for r in rows:
                if not r.action_success or None in r.tokens:
                    continue
                cost = sum(r.tokens)
                if action.startswith('write_'):
                    write_cost[key] = min(write_cost.get(key, cost), cost)
                elif r.after is None:
                    next_stage = 'papers' if action == 'retrieve_more' else 'summary'
                    stale_cost[next_stage] = min(stale_cost.get(next_stage, cost), cost)
        return sorted(f'{stage}/{a}' for (a, stage), cost in write_cost.items()
                      if stage in stale_cost and self.config.max_steps >= (3 if stage == 'summary' else 2)
                      and cost + stale_cost[stage] < self.config.workflow_tokens_budget)

    def _select(self, state, action):
        # 优先同阶段、同篇数的记录；宁可标单题低支持，也不让大量三篇样本冲淡六/九篇成本。
        *sized_keys, stage_key = _keys(state, action, self.calibration_version)
        raw_v2 = self.calibration_version == 'v2' and stage_key[1] == 'papers'
        first = None
        for level, key in zip(LEVELS, sized_keys):
            rows = self.buckets.get(key, ())
            if not rows:
                continue
            count = len({row.review_id for row in rows})
            # A single same-band donor still beats an abundant pool from another band.
            if count >= MIN_REVIEWS or (raw_v2 and level == 'pool_difficulties'):
                return level, rows, count
            if first is None:
                first = level, rows, count
        if first:
            return first
        rows = self.buckets.get(stage_key, ())
        if not rows:
            return 'unsupported', (), 0
        papers = state['retrieved_paper_count']
        distance = min(abs(row.papers_before - papers) for row in rows)
        nearest = [row for row in rows if abs(row.papers_before - papers) == distance]
        return 'nearest_size', nearest, len({row.review_id for row in nearest})

    def __call__(self, before, action_id, rng):
        level, rows, reviews = self._select(before, action_id)
        support = {"calibration_source": self.source, "level": level,
                   "samples": len(rows), "reviews": reviews, "low_support": reviews < MIN_REVIEWS or level == 'nearest_size',
                   "donor_papers_before": None, "donor_step_failed": None}
        if not rows:
            # Censor an unmodelled branch: don't invent a real failure rate,
            # throughput, latency or score. The physical action remains legal.
            support.update(fallback="unsupported_censor_unknown_usage", call_seconds=None,
                           quality_method="not_applicable")
            return Outcome(action_success=False, input_tokens=None, output_tokens=None,
                           verifier_tokens=0, scores=None, simulation_support=support)
        row = rng.choice(rows)
        if self.calibration_version == 'v2' and _stage(before) == 'papers':
            band, donor_band = _coverage_band(_scores(before)), _coverage_band(row.before)
            support.update(coverage_band=band, donor_coverage_band=donor_band, cross_band=band != donor_band)
            support['low_support'] |= band != donor_band
        calls, used = row.call_prefix(before)
        workflow = [c for c in calls if c["role"] != "verify"]
        verifiers = [c for c in calls if c["role"] == "verify"]
        # Local retrieval commits before its first Verify; generation commits
        # only on a successful response (possibly a repair) in the admitted prefix.
        generated = action_id == "retrieve_more" or any(c["status"] == "ok" for c in workflow)
        checked = any(c["status"] == "ok" for c in verifiers)
        accepted = (action_id.startswith("write_") and generated and used is not None
                    and used < before["workflow_tokens_budget"])
        estimate, method = self.quality.estimate(before, action_id) if accepted else (None, "not_applicable")
        support.update(call_seconds=total(c["elapsed_seconds"] for c in calls), quality_method=method,
                       donor_papers_before=row.papers_before, donor_step_failed=not row.step_success)
        return Outcome(action_success=generated, input_tokens=total(c["prompt_tokens"] for c in workflow),
                       output_tokens=total(c["completion_tokens"] for c in workflow),
                       verifier_tokens=total(total((c["prompt_tokens"], c["completion_tokens"])) for c in verifiers),
                       scores=row.transferred_scores(before) if checked else None, estimated_final_quality=estimate,
                       quality_method=method, simulation_support=support, calls=calls)

    def make_env(self, **kwargs):
        """Create an interactive empirical simulator with the captured physical config."""
        return CalibratedEnv(self, **kwargs)

    def manifest(self):
        joints, action_counts = [], dict.fromkeys(ACTIONS, 0)
        for key, rows in sorted(self.buckets.items()):
            if len(key) != 6:
                continue
            action, stage, papers, scope, band, difficulty = key
            joints.append({"action": action, "stage": stage, "papers_before": papers, "valid_dimensions": list(scope),
                           "score_band": band, "difficulty": difficulty, "samples": len(rows),
                           "observed_failed_steps": sum(not r.step_success for r in rows),
                           "reviews": len({r.review_id for r in rows})})
            action_counts[action] += len(rows)
        v2 = self.calibration_version == 'v2'
        return {"source_kind": "empirical_simulation", "calibration_source": self.source,
                'calibration_version': self.calibration_version,
                **({'quality_model': self.quality.manifest(), 'raw_coverage_band_edges': [.40, .70]} if v2 else {}),
                **({"retrieval_spec_sha256": self.retrieval_spec_sha256} if self.mode == "formal" else {}),
                "config": asdict(self.config), "corpus_size": self.corpus_size,
                "training_review_ids": self.reviews.copy(), "review_count": len(self.reviews),
                "episodes": self.episodes, "joint_bins": joints, "joint_bin_count": len(joints),
                "action_samples": action_counts, "min_reviews": MIN_REVIEWS,
                "backoff_levels": [*LEVELS, "nearest_size", "unsupported"], "quality_labels": len(self.quality.targets),
                "quality_label_source": self.quality.source, "unscorable_writes": self.unscorable_writes.copy(),
                "assumptions": [("Raw papers: fresh coverage bands <.40, [.40,.70), >=.70, unknown. Prefer same-size same-band across difficulties even at one review; cross-band draws are low support. Other stages retain v1 bands." if v2 else
                                 "Uniform joint step resampling within pool; bands <.5, <.75, >=.75, unknown."),
                                "同篇数优先；缺该篇数才借同阶段最近篇数，始终标低支持。" if v2 else
                                "同篇数优先选至少两题的最窄桶，否则取同篇数非空桶；完全缺该篇数才借相同阶段的最近篇数，始终标低支持。",
                                "Same action and material stage only. Unsupported => censored unknown usage; not physical illegality or observed failure.",
                                "Clip transferred dimension-wise changes; newly valid dimensions use donor post scores.",
                                "Replay only budget-admitted calls, including repairs and observed cache hits; no request-content cache model. Sum time and cost of the executed prefix only; unknown values stay missing.",
                                *(["Estimated quality: shared fresh-coverage ridge line; review-balanced stage/writer residual shrunk to zero. Clip to [0,1].",
                                   "No fresh input or no fresh fit: review-balanced fit label mean. No fit labels: cost-only."] if v2 else
                                  ["Fresh scores: clipped Verifier + train label residual (>=2 topics), else fresh valid Verifier mean.",
                                   "No fresh scores: review-balanced train label mean for unscored writer/stage (>=2 topics), else pooled review-balanced training mean (low support). Without either signal, cost-only diagnostics only."])]}


def _physical_settings(episodes, manifest, corpus_size):
    # An episode's source_papers are only its retrieved subset, never the corpus.
    if len({(ep['source']['source_id'], ep['source']['corpus_sha256']) for ep in episodes}) != 1:
        raise ValueError('Cannot pool training trajectories from different corpora')
    sizes = {ep['source']['corpus_size'] for ep in episodes}
    if manifest and "corpus_size" in manifest:
        sizes.add(manifest["corpus_size"])
    if corpus_size is not None:
        sizes.add(corpus_size)
    if len(sizes) != 1 or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("Supply one consistent, positive corpus_size for the training corpus")
    size, = sizes
    for ep in episodes:
        for tr in ep["transitions"]:
            for state, mask in ((tr["state_before"], tr["action_mask"]),
                                (tr["state_after"], tr["next_mask"])):
                papers = state["retrieved_paper_count"]
                if papers > size or (not state["terminal"] and mask[0] != (papers < size)):
                    raise ValueError("corpus_size disagrees with recorded retrieval mask/count")
            if tr["action_id"] == "retrieve_more":
                n = tr["state_before"]["retrieved_paper_count"]
                m = tr["state_after"]["retrieved_paper_count"]
                if m != n and m != min(size, n + ep["config"]["retrieval_batch_size"]):
                    raise ValueError("Retrieval increment disagrees with corpus_size/RunnerConfig")
    return size


def validate_calibration_inputs(episodes, labels, *, mode, manifest, corpus_size=None):
    """Check the full batch (including any holdout) without fitting or simulating."""
    episodes = list(episodes)
    joined = validate_collection(episodes, labels, mode=mode, manifest=manifest)
    size = _physical_settings(episodes, manifest, corpus_size)
    fingerprint = None
    if mode == 'formal':
        fingerprint = json_sha256(BilingualRetriever.spec())
        for ep in episodes:
            retrieval = ep.get('retrieval')
            if not isinstance(retrieval, dict) or json_sha256(retrieval.get('spec')) != fingerprint:
                raise ValueError(f"{ep['trajectory_id']}: missing or incompatible frozen retrieval.spec")
    return joined, size, fingerprint


def fit_calibration(episodes, labels=(), *, mode="formal", manifest=None, corpus_size=None, calibration_version='v1'):
    """Fit validated training data only; select the supervised coverage model with 'v2'."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError("No training episodes")
    joined, size, fingerprint = validate_calibration_inputs(episodes, labels, mode=mode, manifest=manifest,
                                                            corpus_size=corpus_size)
    return Calibration(episodes, joined, mode, size, fingerprint, calibration_version)


class CalibratedEnv(SyntheticEnv):
    def __init__(self, model, **kwargs):
        super().__init__(model.config, outcome_model=model, corpus_size=model.corpus_size,
                         source='calibrated', **kwargs)
        if self.reward_config.quality_weight:
            if model.calibration_version == 'v2' and model.quality.pooled_mean is None:
                raise ValueError('Supervised v2 needs training labels; collect labels or use quality_weight=0')
            if model.unscorable_writes:
                raise ValueError('Missing training labels for possible writes without fresh Verifier scores: '
                                 + ', '.join(model.unscorable_writes) + '; collect labels or use quality_weight=0')
        self.model = model

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if options.get('corpus_size', self.corpus_size) != self.corpus_size:
            raise ValueError('Cannot change the fitted corpus size')
        if 'difficulty' not in options:
            if seed is not None:
                self._rng.seed(seed)
            topic = self._rng.choice(self.model.reviews)
            options.update(difficulty=self.model.difficulties[topic])
            options.setdefault('review_id', 'sim-' + topic)
            seed = None
        return super().reset(seed=seed, options=options)


def _summary(rows):
    costs = [r['tokens'] for r in rows if r['tokens'] is not None]
    deltas = [d for r in rows for d in r['deltas'] if d is not None]
    return {'episodes': len(rows), 'completion_rate': fmean(r['done_ok'] for r in rows) if rows else None,
            'token_median': median(costs) if costs else None, 'known_cost_episodes': len(costs),
            'mean_steps': fmean(r['steps'] for r in rows) if rows else None,
            'mean_comparable_delta': fmean(deltas) if deltas else None, 'comparable_steps': len(deltas)}


def _rule_detail(rows):
    known = [r['tokens'] for r in rows if r['tokens'] is not None]
    # Beyond one retrieve/summarize/write cycle. Exact paths/lengths remain available.
    long = [r for r in rows if r['steps'] > 3]
    return {**_summary(rows), 'mean_known_tokens': fmean(known) if known else None,
            'terminal_reasons': dict(Counter(r['terminal_reason'] for r in rows)),
            'long_path_min_steps': 4, 'long_path_episodes': len(long),
            'long_path_known_cost_episodes': sum(r['tokens'] is not None for r in long),
            'runs': [{k: r[k] for k in ('rule_trajectory_id', 'actions', 'steps', 'terminal_reason', 'tokens')}
                     for r in rows]}


def evaluate_holdout(episodes, labels=(), *, holdout_review_ids, mode='formal', manifest=None, seed=0, repeats=3,
                     calibration_version='v1'):
    """按综述整组留出。只对同一 Rule 统计动力学；所有留出的成功标签可验证终稿代理。"""
    from rl.reward import RewardConfig
    episodes, labels = list(episodes), list(labels)
    joined, size, fingerprint = validate_calibration_inputs(episodes, labels, mode=mode, manifest=manifest)
    ids = {e['review_id'] for e in episodes}
    holdout = set(holdout_review_ids)
    if not holdout or not holdout < ids or type(repeats) is not int or repeats < 1:
        raise ValueError('Need a nonempty proper review holdout and positive repeat count')
    train = [e for e in episodes if e['review_id'] not in holdout]
    held = [e for e in episodes if e['review_id'] in holdout]
    train_labels = {e['trajectory_id']: joined[e['trajectory_id']] for e in train if e['trajectory_id'] in joined}
    model = Calibration(train, train_labels, mode, size, fingerprint, calibration_version)
    if any(e['config'] != asdict(model.config) for e in held):
        raise ValueError('Validation Runner config differs from fitted dynamics')
    env = model.make_env(reward_config=RewardConfig(quality_weight=0))
    rule = RuleScheduler(model.config)
    actual, simulated, manuscripts = [], [], []
    low_support = sim_steps = 0
    observed_failed = unsupported = size_backoffs = cross_band_steps = supported_failed = 0
    baseline = fmean(model.quality.targets) if model.quality.targets else None
    review_mean = model.quality.pooled_mean
    for ep in held:
        tid, rid = ep['trajectory_id'], ep['review_id']
        tag = joined.get(tid)
        if tag:
            tr = ep['transitions'][-1]
            state, writer = tr['state_before'], tr['action_id']
            y = tag['overall_score']
            estimate, method = model.quality.estimate(state, writer)
            manuscripts.append({'trajectory_id': tid, 'review_id': rid, 'policy': ep['policy'],
                                'stage': _stage(state), 'writer': writer, 'coverage': _scores(state).get('coverage'),
                                'label_source': tag['source_kind'], 'overall_score': y,
                                'estimated_final_quality': estimate, 'quality_method': method,
                                'abs_error': abs(estimate - y) if estimate is not None else None,
                                'train_label_mean': baseline, 'train_review_mean': review_mean,
                                'label_mean_abs_error': abs(baseline - y) if baseline is not None else None,
                                'review_mean_abs_error': abs(review_mean - y) if review_mean is not None else None})
        if ep['policy'] != 'rule':
            continue
        observed_failed += sum(not t['success'] for t in ep['transitions'])
        actual.append({'review_id': rid, 'rule_trajectory_id': tid, 'terminal_reason': ep['terminal_reason'],
                       'actions': [t['action_id'] for t in ep['transitions']],
                       'done_ok': ep['success'], 'tokens': ep['total_workflow_tokens'], 'steps': ep['step_count'],
                       'deltas': [t['verifier_delta'] for t in ep['transitions']]})
        for _ in range(repeats):
            state, info = env.reset(seed=seed + len(simulated), options={'review_id': ep['review_id'],
                                    'difficulty': ep['public_task']['difficulty']})
            deltas, actions = [], []
            while not state['terminal']:
                a = rule.select_action(state, info['action_mask'])
                actions.append(a)
                state, _, _, _, info = env.step(a)
                step = info['transition']
                deltas.append(step['verifier_delta'])
                sim_steps += 1
                support = step['simulation_support']
                low_support += int(support['low_support'])
                unsupported += int(support['level'] == 'unsupported')
                size_backoffs += int(support['level'] == 'nearest_size')
                cross_band_steps += int(support.get('cross_band', False))
                supported_failed += int(not step['success'] and support['level'] != 'unsupported')
            simulated.append({'review_id': rid, 'rule_trajectory_id': tid, 'actions': actions,
                              'terminal_reason': state['terminal_reason'], 'done_ok': state['terminal_reason'] == 'final_written',
                              'steps': len(deltas), 'tokens': state['workflow_tokens_used'], 'deltas': deltas})
    compared = [r for r in manuscripts if r['abs_error'] is not None]
    baseline_rows = [r for r in compared if r['train_label_mean'] is not None]
    return {'mode': mode, 'calibration_version': calibration_version, 'fit_review_ids': model.reviews,
            'holdout_review_ids': sorted(holdout), 'seed': seed, 'rule_repeats_per_episode': repeats,
            'rule_by_review': {rid: {name: _rule_detail([r for r in rows if r['review_id'] == rid])
                                     for name, rows in [('observed', actual), ('simulated', simulated)]}
                               for rid in sorted(holdout)},
            'observed_rule': {**_summary(actual), 'observed_failed_steps': observed_failed},
            'simulated_rule': {**_summary(simulated), 'supported_failed_steps': supported_failed,
                               'unsupported_censored_steps': unsupported, 'nearest_size_steps': size_backoffs,
                               'raw_cross_band_steps': cross_band_steps if calibration_version == 'v2' else None},
            'low_support_simulated_steps': low_support, 'simulated_steps': sim_steps,
            'quality': {'fit_labels': len(model.quality.targets), 'held_episodes': len(held),
                        'held_accepted_drafts': sum(e['success'] for e in held), 'held_labels': len(manuscripts),
                        'compared': len(compared), 'compared_reviews': len({r['review_id'] for r in compared}),
                        'mae': fmean(r['abs_error'] for r in compared) if compared else None,
                        'baseline_compared': len(baseline_rows),
                        'train_mean_baseline_mae': fmean(r['label_mean_abs_error'] for r in baseline_rows) if baseline_rows else None,
                        'train_review_mean_baseline_mae': fmean(r['review_mean_abs_error'] for r in baseline_rows) if baseline_rows else None,
                        'by_manuscript': manuscripts}}
