"""Empirical simulation from validated *training* Runner episodes (CONTRACT §5).

    model = fit_calibration(episodes, labels, mode="fixture", corpus_size=12)
    env = model.make_env()  # same reset/step and physical masks as SyntheticEnv

Fixture mode tests the pipeline; it is not evidence of real calibration. No
backend, Judge, artifact write or network operation is performed here.
"""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean, median

from core.rule_scheduler import RuleScheduler
from core.runner import total
from core.schema import ACTIONS, SCORE_KEYS, RunnerConfig, verifier_value
from experiments.collect import validate_collection
from rl.synthetic_env import Outcome, SyntheticEnv


TOKEN_FIELDS = ("input_tokens", "output_tokens", "verifier_tokens")
LEVELS = ("joint", "pool_difficulties", "pool_score_bands", "same_stage")
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


def _keys(state, action):
    scores = _scores(state)
    value = fmean(scores.values()) if scores else None
    band = ("unknown" if value is None else "low" if value < 0.5 else
            "mid" if value < 0.75 else "high")
    base = (action, _stage(state))
    scope = tuple(scores)
    return (base + (scope, band, state["difficulty"]), base + (scope, band),
            base + (scope,), base)


def _clip(value):
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class _Row:
    review_id: str
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
        return cls(tr["review_id"], tokens, executed, _scores(before), checked, tuple(calls))

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


class Calibration:
    """Fitted outcome callable. Instantiate via fit_calibration (T4 provenance gate)."""

    def __init__(self, episodes, labels, mode, corpus_size):
        self.mode = mode
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
                for key in _keys(state, action):
                    self.buckets[key].append(row)
        label_source = "synthetic_fixture" if mode == "fixture" else "train_judge"
        self.quality = _Quality(episodes, labels, label_source)
        self.unscorable_writes = self._unscorable_writes()

    def _unscorable_writes(self):
        if self.quality.pooled_mean is not None:
            return []
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
        # Prefer the narrowest pool supported by independent topics. If no
        # pool has two topics, retain the narrowest nonempty one as low support.
        first = None
        for level, key in zip(LEVELS, _keys(state, action)):
            rows = self.buckets.get(key, ())
            if not rows:
                continue
            count = len({row.review_id for row in rows})
            if count >= MIN_REVIEWS:
                return level, rows, count
            if first is None:
                first = level, rows, count
        return first or ("unsupported", (), 0)

    def __call__(self, before, action_id, rng):
        level, rows, reviews = self._select(before, action_id)
        support = {"calibration_source": self.source, "level": level,
                   "samples": len(rows), "reviews": reviews, "low_support": reviews < MIN_REVIEWS}
        if not rows:
            # Censor an unmodelled branch: don't invent a real failure rate,
            # throughput, latency or score. The physical action remains legal.
            support.update(fallback="unsupported_censor_unknown_usage", call_seconds=None,
                           quality_method="not_applicable")
            return Outcome(action_success=False, input_tokens=None, output_tokens=None,
                           verifier_tokens=0, scores=None, simulation_support=support)
        row = rng.choice(rows)
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
        support.update(call_seconds=total(c["elapsed_seconds"] for c in calls), quality_method=method)
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
            if len(key) != 5:
                continue
            action, stage, scope, band, difficulty = key
            joints.append({"action": action, "stage": stage, "valid_dimensions": list(scope),
                           "score_band": band, "difficulty": difficulty, "samples": len(rows),
                           "reviews": len({r.review_id for r in rows})})
            action_counts[action] += len(rows)
        return {"source_kind": "empirical_simulation", "calibration_source": self.source,
                "config": asdict(self.config), "corpus_size": self.corpus_size,
                "training_review_ids": self.reviews.copy(), "review_count": len(self.reviews),
                "episodes": self.episodes, "joint_bins": joints, "joint_bin_count": len(joints),
                "action_samples": action_counts, "min_reviews": MIN_REVIEWS,
                "backoff_levels": [*LEVELS, "unsupported"], "quality_labels": len(self.quality.targets),
                "quality_label_source": self.quality.source, "unscorable_writes": self.unscorable_writes.copy(),
                "assumptions": ["Uniform joint step resampling within pool; bands <.5, <.75, >=.75, unknown.",
                                "Choose narrowest pool with >=2 topics, else narrowest nonempty pool.",
                                "Same action and material stage only. Unsupported => censored unknown usage; not physical illegality or observed failure.",
                                "Clip transferred dimension-wise changes; newly valid dimensions use donor post scores.",
                                "Replay only budget-admitted calls, including repairs and observed cache hits; no request-content cache model. Sum time and cost of the executed prefix only; unknown values stay missing.",
                                "Fresh scores: clipped Verifier + train label residual (>=2 topics), else fresh valid Verifier mean.",
                                "No fresh scores: review-balanced train label mean for unscored writer/stage (>=2 topics), else pooled review-balanced training mean (low support). Without either signal, cost-only diagnostics only."]}


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


def fit_calibration(episodes, labels=(), *, mode="formal", manifest=None, corpus_size=None):
    """Fit only T4-validated training trajectories; mode='fixture' is opt-in."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError("No training episodes")
    joined = validate_collection(episodes, labels, mode=mode, manifest=manifest)
    size = _physical_settings(episodes, manifest, corpus_size)
    return Calibration(episodes, joined, mode, size)


class CalibratedEnv(SyntheticEnv):
    def __init__(self, model, **kwargs):
        super().__init__(model.config, outcome_model=model, corpus_size=model.corpus_size,
                         source='calibrated', **kwargs)
        if self.reward_config.quality_weight and model.unscorable_writes:
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


def evaluate_holdout(episodes, labels=(), *, holdout_review_ids, mode='formal', manifest=None, seed=0, repeats=3):
    """按综述整组留出。只对同一 Rule 统计动力学；所有留出的成功标签可验证终稿代理。"""
    from rl.reward import RewardConfig
    episodes, labels = list(episodes), list(labels)
    joined = validate_collection(episodes, labels, mode=mode, manifest=manifest)
    _physical_settings(episodes, manifest, None)
    ids = {e['review_id'] for e in episodes}
    holdout = set(holdout_review_ids)
    if not holdout or not holdout < ids or type(repeats) is not int or repeats < 1:
        raise ValueError('Need a nonempty proper review holdout and positive repeat count')
    train = [e for e in episodes if e['review_id'] not in holdout]
    held = [e for e in episodes if e['review_id'] in holdout]
    model = fit_calibration(train, [joined[e['trajectory_id']] for e in train if e['trajectory_id'] in joined],
                            mode=mode, manifest=manifest)
    if any(e['config'] != asdict(model.config) for e in held):
        raise ValueError('Validation Runner config differs from fitted dynamics')
    env = model.make_env(reward_config=RewardConfig(quality_weight=0))
    rule = RuleScheduler(model.config)
    actual, simulated, errors, baseline_errors = [], [], [], []
    low_support = sim_steps = 0
    baseline = fmean(model.quality.targets) if model.quality.targets else None
    for ep in held:
        tag = joined.get(ep['trajectory_id'])
        if tag:
            tr = ep['transitions'][-1]
            estimate, _ = model.quality.estimate(tr['state_before'], tr['action_id'])
            if estimate is not None:
                errors.append(abs(estimate - tag['overall_score']))
                if baseline is not None:
                    baseline_errors.append(abs(baseline - tag['overall_score']))
        if ep['policy'] != 'rule':
            continue
        actual.append({'done_ok': ep['success'], 'tokens': ep['total_workflow_tokens'], 'steps': ep['step_count'],
                       'deltas': [t['verifier_delta'] for t in ep['transitions']]})
        for _ in range(repeats):
            state, info = env.reset(seed=seed + len(simulated), options={'review_id': ep['review_id'],
                                    'difficulty': ep['public_task']['difficulty']})
            deltas = []
            while not state['terminal']:
                a = rule.select_action(state, info['action_mask'])
                state, _, _, _, info = env.step(a)
                step = info['transition']
                deltas.append(step['verifier_delta'])
                sim_steps += 1
                low_support += int(step['simulation_support']['low_support'])
            simulated.append({'done_ok': state['terminal_reason'] == 'final_written', 'steps': len(deltas),
                              'tokens': state['workflow_tokens_used'], 'deltas': deltas})
    return {'mode': mode, 'fit_review_ids': model.reviews, 'holdout_review_ids': sorted(holdout),
            'observed_rule': _summary(actual), 'simulated_rule': _summary(simulated),
            'low_support_simulated_steps': low_support, 'simulated_steps': sim_steps,
            'quality': {'held_labels': sum(e['trajectory_id'] in joined for e in held), 'compared': len(errors),
                        'mae': fmean(errors) if errors else None,
                        'train_mean_baseline_mae': fmean(baseline_errors) if baseline_errors else None}}
