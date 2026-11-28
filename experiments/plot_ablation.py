"""从 experiments.ablate 的 comparison.json 与 15 个 result.json 制作静态图表，不训练或访问轨迹。

python -m experiments.plot_ablation --input RUN_DIR --output NEW_REPORT_DIR
输出 SVG + PNG、UTF-8 CSV；null 表示缺失/未达标，SD 为跨训练种子的样本标准差。
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import fmean, stdev
from xml.sax.saxutils import escape

from core.schema import ACTIONS


GROUPS = {'base': '共享基准', 'buffer_small': '小回放池', 'buffer_large': '大回放池',
          'target_frequent': '频繁同步', 'target_rare': '稀疏同步'}
ACTION_LABELS = {'retrieve_more': '检索', 'summarize_fast': '快速摘要', 'summarize_deep': '深度摘要',
                 'write_fast': '快速写作', 'write_deep': '深度写作'}
# dataviz Light / Dark slot 1–3, all-pairs validated; panel headings and CSVs also identify every series.
LIGHT = ('#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#2a78d6', '#eb6834', '#1baf7a')
DARK = ('#1a1a19', '#ffffff', '#c3c2b7', '#383835', '#3987e5', '#d95926', '#199e70')


def require(valid, message):
    if not valid:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def packed(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def csv_table(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'null' if v is None else v for k, v in row.items()})


def ratio(a, b):
    return a / b if a is not None and b else None


def stats(values):
    known = [v for v in values if v is not None]
    return (fmean(known) if known else None, stdev(known) if len(known) > 1 else None, len(known))


def seed_stats(rows, key):
    mean, sd, n = stats([r[key] for r in rows])
    return {f'{key}_seed_mean': mean, f'{key}_seed_sd': sd, f'{key}_valid_seeds': n}


def counted(rows, key):
    vals = [r[key] for r in rows]
    return None if None in vals else sum(vals)


def load(root):
    comparison = read(root / 'comparison.json')
    require(comparison.keys() == GROUPS.keys(), 'Expected all five ablation configurations in comparison.json')
    runs = {}
    for name, group in comparison.items():
        seeds = group['seeds']
        require(len(seeds) == len(set(seeds)) == 3 and all(type(s) is int for s in seeds),
                f'{name}: expected three distinct integer seeds')
        batch = [read(root / f'{name}-seed{s}' / 'result.json') for s in seeds]
        for s, r in zip(seeds, batch):
            require((r['seed'], r['name']) == (s, name), f'{name}/{s}: result identity mismatch')
            steps = [e['step'] for e in r['evaluations']]
            require(steps and steps == sorted(set(steps)) and steps[0] > 0 and steps[-1] == r['config']['steps'],
                    f'{name}/{s}: incomplete or unordered evaluation grid')
        require([r['evaluations'][-1] for r in batch] == group['last_evaluation_by_seed'] and
                [r['steps_to_converge'] for r in batch] == group['steps_to_converge_by_seed'],
                f'{name}: result.json and comparison.json disagree')
        final_mean, final_sd, _ = stats([r['evaluations'][-1]['mean_return'] for r in batch])
        for x, y in ((final_mean, group['last_mean_return']), (final_sd, group['last_return_sd'])):
            require(x == y or (x is not None and y is not None and math.isclose(x, y)),
                    f'{name}: final return mean/SD differs from comparison.json')
        runs[name] = batch
    reference = runs['base'][0]
    grid = [e['step'] for e in reference['evaluations']]
    for name, batch in runs.items():
        require({r['seed'] for r in batch} == {r['seed'] for r in runs['base']}, f'{name}: seed set differs')
        for r in batch:
            require(r['data_source'] == reference['data_source'] and r['threshold'] == .5 and
                    r['consecutive_evaluations'] == 3 and r['config'] == batch[0]['config'] and
                    [e['step'] for e in r['evaluations']] == grid and
                    r['evaluations'][-1]['episodes'] == reference['evaluations'][-1]['episodes'] > 0,
                    f"{name}/{r['seed']}: unequal source, settings, steps or final episode counts")
    return runs


CATEGORIES = ('ineffective', 'redundant', 'premature_write')
CATEGORY_COUNTS = ('count', 'applicable_steps', 'known_steps', 'unknown_steps')


def summarize(runs):
    tables = {key: [] for key in ('curve_seed', 'curve_summary', 'final_seed', 'final_summary',
                                   'actions_seed', 'actions_summary', 'mismatch_categories', 'mismatch_categories_summary')}
    for group in GROUPS:
        batch = runs[group]
        source = batch[0]['data_source']
        key = {'data_source': source, 'config': group}
        seed_rows, action_rows = [], []
        for evaluations in zip(*(r['evaluations'] for r in batch)):
            values = [e['mean_return'] for e in evaluations]
            mean, sd, n = stats(values)
            for r, e in zip(batch, evaluations):
                tables['curve_seed'].append({**key, 'seed': r['seed'], 'step': e['step'],
                    'mean_return': e['mean_return'], 'episodes': e['episodes'], 'returns_json': packed(e['returns'])})
            # Only draw a point/band when all three seed means exist. Gaps are not interpolated.
            tables['curve_summary'].append({**key, 'step': evaluations[0]['step'], 'valid_seeds': n,
                'mean_return': mean if n == 3 else None, 'sd': sd if n == 3 else None,
                'min': min(values) if n == 3 else None, 'max': max(values) if n == 3 else None})
        for r in batch:
            e = r['evaluations'][-1]
            ident = {**key, 'seed': r['seed'], 'step': e['step']}
            steps, episodes, known_cost = e['steps'], e['episodes'], e['known_cost_episodes']
            require(0 <= known_cost <= episodes and (e['mean_known_episode_tokens'] is None) == (known_cost == 0),
                    f'{ident}: invalid known-cost denominator')
            mismatch = e.get('mismatch')  # Pre-classifier files carry mismatch_status, not all-zero counts.
            m = mismatch if mismatch is not None else dict.fromkeys(('any_count', 'known_steps', 'unknown_steps',
                    'per_episode', 'execution_failure_count', 'unsupported_censored_steps'))
            if mismatch is not None:
                require(m['known_steps'] + m['unknown_steps'] == steps and 0 <= m['any_count'] <= m['known_steps'] and
                        0 <= m['unsupported_censored_steps'] <= m['unknown_steps'] and
                        math.isclose(m['per_episode'], m['any_count'] / episodes) and set(m['by_category']) == set(CATEGORIES),
                        f'{ident}: wrong mismatch denominator or categories')
            seed_rows.append({**ident, 'capacity': r['config']['capacity'], 'target_interval': r['config']['target_interval'],
                'steps_to_converge': r['steps_to_converge'], 'mean_return': e['mean_return'], 'episodes': episodes,
                'steps': steps, 'completion_rate': e['completion_rate'], 'mean_known_episode_tokens': e['mean_known_episode_tokens'],
                'known_cost_episodes': known_cost, 'unknown_cost_steps': e['unknown_cost_steps'], 'any_count': m['any_count'],
                'known_steps': m['known_steps'], 'unknown_steps': m['unknown_steps'], 'mismatch_per_episode': m['per_episode'],
                'unknown_step_share': ratio(m['unknown_steps'], steps), 'execution_failure_count': m['execution_failure_count'],
                'unsupported_censored_steps': m['unsupported_censored_steps'], 'mismatch_status': e.get('mismatch_status')})
            for category in CATEGORIES:
                c = m['by_category'][category] if mismatch is not None else dict.fromkeys(CATEGORY_COUNTS)
                if mismatch is not None:
                    require(c['known_steps'] + c['unknown_steps'] == c['applicable_steps'] and 0 <= c['count'] <= c['known_steps'],
                            f'{ident}/{category}: wrong applicable denominator')
                tables['mismatch_categories'].append({**ident, 'category': category, **{k: c[k] for k in CATEGORY_COUNTS},
                                                       'count_per_known_step': ratio(c['count'], c['known_steps'])})
            require(set(e['actions']) == set(ACTIONS) and sum(a['calls'] for a in e['actions'].values()) == steps,
                    f'{ident}: five action counts must sum to evaluation steps')
            for a in ACTIONS:
                v = e['actions'][a]
                action_rows.append({**ident, 'action': a, 'steps': steps, 'calls': v['calls'],
                    'call_share': ratio(v['calls'], steps), 'opportunities': v['opportunities'],
                    'selection_rate': v['selection_rate'], 'token_share': v['token_share']})
        tables['final_seed'].extend(seed_rows)
        tables['actions_seed'].extend(action_rows)
        totals = {k: counted(seed_rows, k) for k in ('episodes', 'steps', 'known_cost_episodes', 'unknown_cost_steps',
                                                     'any_count', 'known_steps', 'unknown_steps', 'execution_failure_count',
                                                     'unsupported_censored_steps')}
        known_tokens = sum(s['mean_known_episode_tokens'] * s['known_cost_episodes'] for s in seed_rows if s['known_cost_episodes'])
        final = {**key, 'step': seed_rows[0]['step'], 'capacity': seed_rows[0]['capacity'],
                 'target_interval': seed_rows[0]['target_interval'], 'seeds': len(batch),
                 'steps_to_converge_by_seed': packed({s['seed']: s['steps_to_converge'] for s in seed_rows}),
                 **totals, 'mean_tokens_per_known_episode': ratio(known_tokens, totals['known_cost_episodes']),
                 'unknown_step_share_pooled': ratio(totals['unknown_steps'], totals['steps'])}
        for k in ('mean_return', 'completion_rate', 'mean_known_episode_tokens', 'mismatch_per_episode', 'unknown_step_share'):
            final.update(seed_stats(seed_rows, k))
        tables['final_summary'].append(final)
        for category in CATEGORIES:
            rows = [r for r in tables['mismatch_categories'] if r['category'] == category and r['config'] == group]
            c = {k: counted(rows, k) for k in CATEGORY_COUNTS}
            tables['mismatch_categories_summary'].append({**key, 'step': final['step'], 'category': category, **c,
                'count_per_known_step_pooled': ratio(c['count'], c['known_steps']), **seed_stats(rows, 'count_per_known_step')})
        for a in ACTIONS:
            rows = [s for s in action_rows if s['action'] == a]
            calls, opp = counted(rows, 'calls'), counted(rows, 'opportunities')
            summary = {**key, 'step': final['step'], 'action': a, 'calls': calls, 'steps': totals['steps'],
                       'call_share_pooled': ratio(calls, totals['steps']), 'opportunities': opp,
                       'selection_rate_pooled': ratio(calls, opp)}
            for k in ('call_share', 'selection_rate', 'token_share'):
                summary.update(seed_stats(rows, k))
            tables['actions_summary'].append(summary)
    return tables


class Sheet:
    """Static 16:9 slide; draw in logical units, render a 1920×1080 opaque PNG."""
    def __init__(self, palette, title, subtitle, edition, number):
        self.bg, self.ink, self.sub, self.grid, *self.colors = palette
        self.svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1440 810"'
                    ' role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title>',
                    f'<desc id="desc">{escape(subtitle)} · {escape(edition)}</desc>',
                    '<g font-family="Helvetica Neue, PingFang SC, sans-serif">']
        self.rect(0, 0, 1440, 810, self.bg)
        self.text(56, 49, edition, 18, secondary=True)
        self.text(1384, 49, f'DQN  /  {number:02d}', 18, anchor='end', secondary=True)
        self.text(56, 108, title, 44, bold=True)
        self.text(56, 149, subtitle, 23, secondary=True)
        self.line(56, 174, 1384, 174, self.grid)

    def text(self, x, y, value, size=20, *, anchor='start', bold=False, secondary=False):
        self.svg.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                        f'font-weight="{600 if bold else 400}" fill="{self.sub if secondary else self.ink}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color, width=1, dashed=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                        f'stroke="{color}" stroke-width="{width}"{dash} />')

    def rect(self, x, y, w, h, color, rx=0):
        self.svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{color}"/>')

    def dot(self, x, y, color):
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="{self.bg}" stroke-width="2"/>')

    def path(self, points, color, band=False):
        points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        if band:
            self.svg.append(f'<polygon points="{points}" fill="{color}" fill-opacity="0.17"/>')
        else:
            self.svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" '
                            'stroke-linejoin="round" stroke-linecap="round"/>')

    def finish(self, note):
        self.line(56, 752, 1384, 752, self.grid)
        self.text(56, 787, note, 19, secondary=True)
        return '\n'.join(self.svg + ['</g></svg>'])


def nice_axis(lo, hi, intervals=4):
    raw = (hi - lo) / intervals
    scale = 10 ** math.floor(math.log10(raw))
    unit = next(f * scale for f in (1, 2, 2.5, 5, 10) if f * scale >= raw)
    return [i * unit for i in range(math.floor(lo / unit), math.ceil(hi / unit) + 1)]


def show(v, fmt='.2f'):
    return 'NA' if v is None else format(v, fmt)


def convergence(t, palette, edition):
    sheet = Sheet(palette, '训练过程中，验证回报如何变化？',
                  '每个点代表同一检查点的 3 个训练种子均值；色带为种子最小至最大值，不是置信区间。', edition, 1)
    rows = t['curve_summary']
    final = {r['config']: r for r in t['final_summary']}
    ticks = nice_axis(min([0] + [r['min'] for r in rows if r['min'] is not None]),
                      max([1] + [r['max'] for r in rows if r['max'] is not None]))
    maxstep = max(r['step'] for r in rows)
    for i, (key, title, groups) in enumerate((('capacity', 'A   回放池容量（经验条数）', ('buffer_small', 'base', 'buffer_large')),
                                               ('target_interval', 'B   目标网络同步间隔（梯度更新次数）', ('target_frequent', 'base', 'target_rare')))):
        top = 216 + i * 265
        sheet.text(56, top, title, 24, bold=True)
        for j, group in enumerate(groups):
            color = sheet.colors[(1, 0, 2)[j]]
            x0, y0 = 105 + j * 451, top + 59
            w, h = 366, 145
            series = [r for r in rows if r['config'] == group]
            X = lambda x: x0 + w * x / maxstep
            Y = lambda y: y0 + h * (ticks[-1] - y) / (ticks[-1] - ticks[0])
            sheet.line(x0, top+32, x0+20, top+32, color, 4)
            sheet.text(x0+28, top+39, f"{GROUPS[group]}  {final[group][key]:,}", 21, bold=True)
            sheet.text(x0+w, top+39, '末次 ' + show(series[-1]['mean_return']), 18, anchor='end', secondary=True)
            for tick in ticks:
                sheet.line(x0, Y(tick), x0+w, Y(tick), sheet.grid)
                sheet.text(x0-9, Y(tick)+5, f'{tick:g}', 15, anchor='end', secondary=True)
            sheet.line(x0, Y(.5), x0+w, Y(.5), sheet.sub, 1.5, dashed=True)
            for tick in (0, maxstep // 2, maxstep):
                sheet.text(X(tick), y0+h+22, f'{tick:,}', 16, anchor='middle', secondary=True)
            segment = []
            for point in [*series, None]:
                if point is not None and point['mean_return'] is not None:
                    segment.append(point)
                    continue
                if segment:
                    coords = lambda field: [(X(r['step']), Y(r[field])) for r in segment]
                    sheet.path(coords('max') + coords('min')[::-1], color, band=True)
                    sheet.path(coords('mean_return'), color)
                    sheet.dot(*coords('mean_return')[-1], color)
                    segment = []
    return sheet.finish('横轴：训练环境步数    纵轴：验证平均回报    虚线：0.50 门槛（连续 3 次达到才算收敛）    同一基准在两行重复展示；缺值断线。')


def mismatch_figure(t, palette, edition):
    rows = t['final_summary']
    step = rows[0]['step']
    sheet = Sheet(palette, '末次验证中，每局有几步可确认错配？',
                  f'训练第 {step:,} 步 · 点为种子均值，横线为种子范围。右侧列出 3 个种子合计的计数和分母。', edition, 2)
    seedrows = t['final_seed']
    vals = [s['mismatch_per_episode'] for s in seedrows if s['mismatch_per_episode'] is not None]
    ticks = nice_axis(0, max([1] + vals))
    x0, w = 255, 430
    X = lambda v: x0 + w * v / ticks[-1]
    sheet.text(56, 233, '配置', 21, bold=True)
    sheet.text(x0, 233, '已确认错配步 / 局', 21, bold=True)
    for x, label in ((865, '错配步 / 局数'), (1050, '可判步'), (1200, '未知步'), (1384, '动作总步')):
        sheet.text(x, 233, label, 21, anchor='end', bold=True)
    for tick in ticks:
        sheet.line(X(tick), 257, X(tick), 642, sheet.grid)
        sheet.text(X(tick), 674, f'{tick:g}', 18, anchor='middle', secondary=True)
    for i, g in enumerate(rows):
        group, y = g['config'], 292 + i * 77
        sheet.text(56, y+8, GROUPS[group], 24, bold=(group == 'base'))
        values = [s['mismatch_per_episode'] for s in seedrows if s['config'] == group]
        n = g['mismatch_per_episode_valid_seeds']
        if n == 3:
            sheet.line(X(min(values)), y, X(max(values)), y, sheet.colors[0], 3)
            for v in (min(values), max(values)):
                sheet.line(X(v), y-7, X(v), y+7, sheet.colors[0], 2)
            x = X(g['mismatch_per_episode_seed_mean'])
            sheet.dot(x, y, sheet.colors[0])
            sheet.text(x, y-12, show(g['mismatch_per_episode_seed_mean']), 18, anchor='middle')
        else:
            sheet.text(x0+6, y+7, f'NA（有效种子 {n}/3）', 19, secondary=True)
        for x, value in ((865, f"{show(g['any_count'], ',d')} / {g['episodes']:,}"), (1050, show(g['known_steps'], ',d')),
                         (1200, show(g['unknown_steps'], ',d')), (1384, f"{g['steps']:,}")):
            sheet.text(x, y+8, value, 23, anchor='end')
    return sheet.finish('分母含全部验证局；未知不等于无错配。多类错配可重叠，总数按步去重；分类计数与分母见 mismatch_categories.csv。')


def resource_figure(t, palette, edition):
    rows = t['final_summary']
    sheet = Sheet(palette, '五种动作如何分配？每局消耗多少？',
                  '动作占比按 3 个种子的调用次数合并计算；每列细条的满量程均为 100%。成本和完成率使用各自的局数分母。', edition, 3)
    centers = [278 + j * 135 for j in range(5)]
    sheet.text(56, 234, '配置', 21, bold=True)
    for x, action in zip(centers, ACTIONS):
        sheet.text(x, 234, ACTION_LABELS[action], 21, anchor='middle', bold=True)
    for x, label in ((938, '总调用'), (1160, '平均 Token / 局'), (1384, '完成率')):
        sheet.text(x, 234, label, 21, anchor='end', bold=True)
    sheet.text(1160, 263, '仅成本已知局', 17, anchor='end', secondary=True)
    sheet.text(1384, 263, '每种子等权', 17, anchor='end', secondary=True)
    group_actions = {(v['config'], v['action']): v for v in t['actions_summary']}
    for i, g in enumerate(rows):
        group, y = g['config'], 312 + i * 78
        sheet.line(56, y-29, 1384, y-29, sheet.grid)
        sheet.text(56, y+7, GROUPS[group], 24, bold=(group == 'base'))
        for x, a in zip(centers, ACTIONS):
            v = group_actions[group, a]['call_share_pooled']
            sheet.text(x, y+5, show(v, '.1%'), 22, anchor='middle')
            sheet.rect(x-42, y+17, 84, 5, sheet.grid)
            if v:
                sheet.rect(x-42, y+17, 84*v, 5, sheet.colors[0])
        sheet.text(938, y+7, f"{g['steps']:,}", 24, anchor='end')
        sheet.text(1160, y+3, show(g['mean_tokens_per_known_episode'], ',.0f'), 24, anchor='end')
        sheet.text(1160, y+26, f"{g['known_cost_episodes']:,} / {g['episodes']:,} 局", 17, anchor='end', secondary=True)
        sheet.text(1384, y+3, show(g['completion_rate_seed_mean'], '.1%'), 24, anchor='end')
        sheet.text(1384, y+26, f"{g['episodes']:,} 局", 17, anchor='end', secondary=True)
    sheet.text(56, 720, '成本＝已知局的工作流 Token 总和 ÷ 已知局数；缺失不按零处理。完成＝终止原因是 final_written。', 20, secondary=True)
    return sheet.finish('完整动作表 actions_summary.csv 同列三种不同口径：占总调用、占可执行机会、占 Token。各自分母不同，不可互换。')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='包含 comparison.json 的运行目录')
    p.add_argument('--output', required=True, type=Path, help='全新输出目录，不覆盖已有报告')
    p.add_argument('--theme', choices=('light', 'dark'), default='light', help='静态图的固定底色')
    args = p.parse_args(argv)
    rsvg, sips = shutil.which('rsvg-convert'), shutil.which('sips')
    if not (rsvg or sips):
        p.error('PNG requires an existing rsvg-convert or macOS sips; nothing is installed by this tool')
    runs = load(args.input)
    tables = summarize(runs)
    source = runs['base'][0]['data_source']
    provenance = '训练轨迹标定的模拟器' if source == 'formal_train_runner' else '模拟演练 · 非正式结果'
    edition = '离线 DQN   /   训练内模拟验证   /   ' + provenance + '   ·   非真实 API 评测'
    palette = LIGHT if args.theme == 'light' else DARK
    figures = {k: f(tables, palette, edition) for k, f in (('convergence', convergence), ('mismatch', mismatch_figure),
                                                           ('resource_use', resource_figure))}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        csv_table(out / f'{name}.csv', rows)
    for name, content in figures.items():
        svg, png = out / f'{name}.svg', out / f'{name}.png'
        svg.write_text(content, encoding='utf-8')
        cmd = [rsvg, '-o', str(png), str(svg)] if rsvg else [sips, '-s', 'format', 'png', str(svg), '--out', str(png)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f'{out}: 3 figures (SVG + PNG), {len(tables)} CSV tables; data_source={source}')


if __name__ == '__main__':
    main()
