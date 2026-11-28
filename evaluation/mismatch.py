"""Post-hoc operational labels for Runner and simulated transitions (CONTRACT §4.2)."""

from math import isclose

from core.schema import ACTIONS, SCORE_KEYS, comparable_delta, verifier_value


MISMATCH_RULES = {
    'version': 'mismatch-v1',
    'quality_threshold': .75, 'retrieval_coverage_threshold': .85, 'low_gain_threshold': .02,
    'premature_min_steps_left_including_write': 2, 'premature_min_tokens_left': 12000,
    'premature_write': 'Quality risk proxy, not final quality. Without a summary only fresh coverage is scored.',
    'per_episode': 'Confirmed mismatched steps / all episodes; unknown steps may hide additional mismatches.',
    'unsupported': 'No-sample censoring is unknown, never an execution failure or a confirmed mismatch.',
}
CATEGORIES = ('ineffective', 'redundant', 'premature_write')


def _any(values):
    return True if True in values else None if None in values else False


def _low_gain(delta):
    if delta is None:
        return None
    threshold = MISMATCH_RULES['low_gain_threshold']
    # Subtracting two decimal scores can put an exact boundary just above .02.
    return delta <= threshold or isclose(delta, threshold, rel_tol=0, abs_tol=1e-12)


def _fresh_scores(state):
    verifier = state['verifier']
    if verifier['checked_material_revision'] != state['material_revision']:
        return {}
    keys = SCORE_KEYS if state['summary_available'] else ('coverage',)
    return {k: verifier['scores'][k] for k in keys if verifier['valid_scores'][k]}


def _redundant_rework(action, before, after):
    """Compare individual target dimensions, not a mean that could hide a useful gain."""
    if any(v < MISMATCH_RULES['quality_threshold'] for v in before.values()):
        return False
    if action == 'retrieve_more':
        if 'coverage' not in before:
            return None
        if before['coverage'] < MISMATCH_RULES['retrieval_coverage_threshold']:
            return False
        targets = ('coverage',)  # Retrieval rechecks coverage only, including after a summary.
    else:
        if not before or before.keys() != after.keys():
            return None
        targets = before
    if not all(k in after for k in targets):
        return None
    return _low_gain(max(after[k] - before[k] for k in targets))


def _premature_write(state):
    used = state['workflow_tokens_used']
    if (state['max_steps'] - state['step_count'] < MISMATCH_RULES['premature_min_steps_left_including_write']
            or used is not None and state['workflow_tokens_budget'] - used < MISMATCH_RULES['premature_min_tokens_left']):
        return False
    scores = _fresh_scores(state)
    if scores and min(scores.values()) >= MISMATCH_RULES['quality_threshold']:
        return False
    return True if scores and used is not None else None


def classify_step(transition):
    """Sparse by_category: absent = inapplicable; None = applicable but unknown. Never mutates input."""
    action, before, after = (transition[k] for k in ('action_id', 'state_before', 'state_after'))
    if action not in ACTIONS:
        raise ValueError(f'Unknown action: {action}')
    write = action.startswith('write_')
    rework = not write and (before['retrieved_paper_count'] > 0 if action == 'retrieve_more' else before['summary_available'])
    easy_deep = action == 'summarize_deep' and before['difficulty'] == 'easy'
    censored = (transition.get('simulation_support') or {}).get('level') == 'unsupported'
    failure = None if censored else not transition['success']
    checked = not write and failure is False and after['material_revision'] > before['material_revision']
    delta = comparable_delta(before, after) if checked else None

    ineffective = failure
    if failure is False and rework:
        value = verifier_value(before)
        ineffective = False if value is not None and value >= MISMATCH_RULES['quality_threshold'] else _low_gain(delta)
    labels = {'ineffective': ineffective}

    if action == 'write_deep':
        labels['redundant'] = None  # No same-input fast-write counterfactual in a transition.
    elif rework or easy_deep:
        candidates = []
        if rework:
            candidates.append(_redundant_rework(action, _fresh_scores(before), _fresh_scores(after) if checked else {}))
        if easy_deep:
            candidates.append(_low_gain(delta))
        labels['redundant'] = _any(candidates)
    if write:
        labels['premature_write'] = _premature_write(before)
    if censored:
        labels = dict.fromkeys(labels)  # Not even a pre-write risk is counted on a censored branch.
    return {'by_category': labels, 'mismatch_any': _any(labels.values()),
            'execution_failed': failure, 'unsupported_censored': censored}


def summarize_mismatch(episodes):
    """Each episode is a list of transitions. Censored steps are a subset of unknown steps."""
    by_category = {k: {'count': 0, 'applicable_steps': 0, 'known_steps': 0, 'unknown_steps': 0} for k in CATEGORIES}
    hits = known = unknown = failures = censored = 0
    for steps in episodes:
        for transition in steps:
            row = classify_step(transition)
            label = row['mismatch_any']
            hits += label is True
            known += label is not None
            unknown += label is None
            failures += row['execution_failed'] is True
            censored += row['unsupported_censored']
            for category, result in row['by_category'].items():
                counts = by_category[category]
                counts['applicable_steps'] += 1
                counts['known_steps' if result is not None else 'unknown_steps'] += 1
                counts['count'] += result is True
    return {'any_count': hits, 'known_steps': known, 'unknown_steps': unknown, 'per_episode': hits / len(episodes),
            'by_category': by_category, 'execution_failure_count': failures, 'unsupported_censored_steps': censored}
