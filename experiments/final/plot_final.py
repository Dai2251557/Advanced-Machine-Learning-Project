"""从发布版 results/final/{report,early} 的 CSV 离线重图，不加载模型、Judge、测试题或轨迹。

python -m experiments.final.plot_final --out-dir outputs/replot

默认绘制四张图（--early-only 只绘制训练内早期图）。SVG 保留可编辑文本；PNG 由
macOS 自带 sips 渲染，无 sips 时可用 --svg-only。sources.csv 记录每张图的实际输入。
"""
import argparse
import csv
import math
import os
import shutil
import subprocess
from pathlib import Path
from statistics import fmean
from xml.sax.saxutils import escape

from experiments.plot_ablation import DARK, LIGHT, Sheet, nice_axis, require

from . import BATCH, POLICIES, PUBLIC, new_output

NAMES = ('Rule', 'v1', 'v2')
LAMBDAS = (0., .25, .5, 1.)
TEST = '独立十题   /   真实工作流与独立盲评'
SYNTHETIC = '合成布局测试   /   非实验结果'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def number(raw):
    if raw is None or raw.strip() in ('', 'NA', 'null'):
        return None
    value = float(raw)
    require(math.isfinite(value), f'非有限数值：{raw}')
    return value


def keyed(rows, *fields):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        require(key not in result, f'重复的记录：{key}')
        result[key] = row
    return result


def hover_box(sheet, x, y, w, h, title):
    sheet.svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                     f'fill="none" pointer-events="all" tabindex="0"><title>{escape(title)}</title></rect>')


def key_label(sheet, x, y, color, label, size=21):
    sheet.line(x, y-7, x+22, y-7, color, 4)
    sheet.text(x+30, y, label, size, bold=True)


def quality_chart(sheet, tasks, data):
    # Six aligned numeric columns, each with a short bar on the original 0..1 scale.
    # No jitter, truncated axis, or connecting lines between unordered questions.
    sheet.text(56, 263, '题号', 21, bold=True)
    panels = [('overall_score', 205, 'A   Judge 总体分'), ('coverage', 819, 'B   Coverage：题面要求覆盖度')]
    for field, left, title in panels:
        sheet.text(left, 216, title, 25, bold=True)
        for j, policy in enumerate(POLICIES):
            x = left + 181*j
            key_label(sheet, x, 263, sheet.colors[j], NAMES[j])
            for i, task in enumerate(tasks):
                y = 302 + 40*i
                row = data[task, policy]
                v = number(row[field])
                if v is None:
                    label = '失败' if row['delivered'] == 'False' else 'NA'
                else:
                    label = f'{v:.2f}'
                    sheet.rect(x, y-5, 93, 6, sheet.grid, 2)
                    if v > 0:
                        sheet.rect(x, y-5, 93*v, 6, sheet.colors[j], 2)
                sheet.text(x+153, y+5, label, 21, anchor='end', secondary=v is None)
                hover_box(sheet, x-2, y-14, 159, 29, f'{task} · {NAMES[j]} · {field}：{label}')
    for i, task in enumerate(tasks):
        y = 302 + 40*i
        sheet.line(56, y-19, 1384, y-19, sheet.grid)
        sheet.text(56, y+5, task if len(task) <= 11 else f'题 {i+1:02}', 21, bold=True)
    sheet.text(56, 724, '失败＝未交付终稿；NA＝交付后无有效评分。各小条量程均为 0–1，数值按原始评分保留两位小数。', 20, secondary=True)
    return sheet.finish('数据：report/per_task.csv   ·   总体分为四项评分的等权平均；Coverage 只评公开题面要求是否被回答，不按引文数量计分。')


def cost_chart(sheet, summary):
    means = [number(summary[p,]['mean_workflow_tokens']) for p in POLICIES]
    ticks = nice_axis(0, max([1, *(v for v in means if v is not None)]), 4)
    left, w, right, w2 = 204, 440, 903, 286
    sheet.text(left, 225, 'A   已知成本题的平均工作流 token', 26, bold=True)
    sheet.text(right, 225, 'B   终稿交付率', 26, bold=True)
    for x0, width, grid, lab in [(left, w, ticks, lambda x: f'{x:,.0f}'),
                                   (right, w2, [0, .25, .5, .75, 1], lambda x: f'{x:.0%}')]:
        for tick in grid:
            x = x0 + width*tick/grid[-1]
            sheet.line(x, 281, x, 638, sheet.grid)
            sheet.text(x, 668, lab(tick), 17, anchor='middle', secondary=True)
    for j, policy in enumerate(POLICIES):
        row, y = summary[policy,], 342+118*j
        n, known, done = (int(row[k]) for k in ('n_tasks', 'n_known_cost', 'n_delivered'))
        avg = means[j]
        key_label(sheet, 56, y+5, sheet.colors[j], NAMES[j], 26)
        if avg is not None and avg > 0:
            sheet.rect(left, y-9, w*avg/ticks[-1], 15, sheet.colors[j], 3)
        sheet.text(802, y+6, 'NA' if avg is None else f'{avg:,.0f}', 29, anchor='end', bold=True)
        sheet.text(802, y+34, f'已知 {known}/{n} 题', 18, anchor='end', secondary=True)
        if done:
            sheet.rect(right, y-9, w2*done/n, 15, sheet.colors[j], 3)
        sheet.text(1384, y+6, f'{done/n:.0%}', 29, anchor='end', bold=True)
        sheet.text(1384, y+34, f'交付 {done}/{n} 题', 18, anchor='end', secondary=True)
        hover_box(sheet, left, y-20, w, 40, f'{NAMES[j]}：完整成本 {known}/{n} 题，均值 ' + ('NA' if avg is None else f'{avg:,.2f} token'))
        hover_box(sheet, right, y-20, w2, 40, f'{NAMES[j]}：交付 {done}/{n} 题')
    sheet.text(56, 726, '成本不含 Judge token；失败但用量完整的题仍计成本。不同策略已知成本题可能不同，不能把未知用量当作零。', 19, secondary=True)
    return sheet.finish('数据：report/overview.csv   ·   左：工作流 token 总和 ÷ 成本完整的题数；右：有终稿题数 ÷ 该策略全部 10 道题。')


def utility_chart(sheet, rows):
    data = {}
    for row in rows:
        key = float(row['lambda']), row['policy']
        require(key not in data, f'重复效用记录 {key}')
        data[key] = (int(row['n_common']), number(row['mean_utility']))
    require(set(data) == {(lam, p) for lam in LAMBDAS for p in POLICIES}, '需要三种策略及四档 λ')
    ns = [data[lam, 'rule'][0] for lam in LAMBDAS]
    require(ns[0] >= ns[1] == ns[2] == ns[3], '共同题分母不满足 λ 定义')
    for i, lam in enumerate(LAMBDAS):
        require(all(n == ns[i] and 0 <= n <= 10 and (v is not None) == bool(n)
                    for n, v in [data[lam, p] for p in POLICIES]), '共同题分母或缺失值不一致')
    values = [val for _, val in data.values() if val is not None]
    ticks = nice_axis(min([0, *values]), max([1, *values]), 4)
    x0, y0, w, h = 112, 306, 590, 292
    X = lambda lam: x0 + w*lam
    Y = lambda v: y0 + h*(ticks[-1]-v)/(ticks[-1]-ticks[0])
    sheet.text(56, 226, 'A   成本惩罚与平均效用', 25, bold=True)
    sheet.text(811, 226, 'B   四档逐值对照', 25, bold=True)
    for j in range(3):
        key_label(sheet, 112+118*j, 273, sheet.colors[j], NAMES[j], 20)
    for tick in ticks:
        sheet.line(x0, Y(tick), x0+w, Y(tick), sheet.grid)
        sheet.text(x0-13, Y(tick)+6, f'{tick:g}', 18, anchor='end', secondary=True)
    for lam in LAMBDAS:
        sheet.text(X(lam), y0+h+31, f'{lam:g}', 18, anchor='middle', secondary=True)
    sheet.text(x0+w/2, y0+h+64, '成本权重 λ', 20, anchor='middle')
    for j, policy in enumerate(POLICIES):
        prev = None
        for lam in LAMBDAS:
            n, v = data[lam, policy]
            if v is not None:
                xy = X(lam), Y(v)
                if prev is not None and n == prev[0]:
                    sheet.line(*prev[1], *xy, sheet.colors[j], 2.5)
                sheet.dot(*xy, sheet.colors[j])
                prev = n, xy
            else:
                prev = None
    cols = (814, 975, 1109, 1242, 1376)
    for j, (x, name) in enumerate(zip(cols, ('λ', '共同题', *NAMES))):
        sheet.text(x, 298, name, 22, anchor='start' if j == 0 else 'end', bold=True)
    for i, lam in enumerate(LAMBDAS):
        y = 354+72*i
        row = [data[lam, p][1] for p in POLICIES]
        labels = ['NA' if v is None else f'{v:.3f}' for v in row]
        sheet.line(811, y-32, 1384, y-32, sheet.grid)
        for j, (x, label) in enumerate(zip(cols, (f'{lam:g}', f'{ns[i]}/10', *labels))):
            sheet.text(x, y+6, label, 23, anchor='start' if j == 0 else 'end')
        l = x0-16 if i == 0 else (X(LAMBDAS[i-1])+X(lam))/2
        r = x0+w+16 if i == 3 else (X(LAMBDAS[i+1])+X(lam))/2
        hover_box(sheet, l, y0-8, r-l, h+18, f'λ={lam:g} · 共同题 {ns[i]}/10 · ' + '；'.join(f'{name} {v}' for name, v in zip(NAMES, labels)))
    sheet.text(56, 724, 'λ＝0：同题三方得分均有效；λ＞0：还要求同题三方成本完整。分母变化时不连线；没有共同题显示 NA。', 19, secondary=True)
    return sheet.finish('数据：report/utility_summary.csv   ·   各策略先逐题计算效用，再在同一档的共同题上等权平均。λ 越大，单位工作流成本的扣分越多。')


def final_charts(directory, palette, synthetic=False):
    paths = [directory / name for name in ('per_task.csv', 'overview.csv', 'utility_summary.csv')]
    data = keyed(read_csv(paths[0]), 'review_id', 'policy')
    summary = keyed(read_csv(paths[1]), 'policy')
    tasks = sorted({t for t, _ in data})
    require(len(tasks) == 10 and set(data) == {(t, p) for t in tasks for p in POLICIES}
            and set(summary) == {(p,) for p in POLICIES}, '需要十道独立题的全部三策略记录')
    for p in POLICIES:
        rows = [data[t, p] for t in tasks]
        require(all(r['delivered'] in ('True', 'False') for r in rows), 'delivered 不是 True/False')
        for row in rows:
            for k in ('overall_score', 'coverage'):
                v = number(row[k])
                require(v is None or (row['delivered'] == 'True' and 0 <= v <= 1), f'无效评分：{row["review_id"]}/{p}/{k}')
        agg = summary[p,]
        n, known, done = (int(agg[k]) for k in ('n_tasks', 'n_known_cost', 'n_delivered'))
        mean = number(agg['mean_workflow_tokens'])
        require(n == 10 and sum(r['delivered'] == 'True' for r in rows) == done and
                sum(number(r['workflow_tokens']) is not None for r in rows) == known and
                ((known == 0 and mean is None) or (known > 0 and mean is not None and mean >= 0)), '交付或已知成本的分母不一致')
    edition = SYNTHETIC if synthetic else TEST
    return [
        ('01_quality_by_task', quality_chart(Sheet(palette, '逐题比较：三种策略的总体分与覆盖度',
             '同一行是同一道题，题号顺序固定；左右分别读数，不按得分重新排序。', edition, 1), tasks, data), paths[:1]),
        ('02_cost_and_delivery', cost_chart(Sheet(palette, '每种策略花多少工作流 token，交付多少终稿？',
             '三种策略各发起 10 题；比较单位成本时只使用记录完整的题，计算交付率时保留全部失败题。', edition, 2), summary), paths[1:2]),
        ('03_utility_by_lambda', utility_chart(Sheet(palette, '把工作流成本计入后，三种策略的效用如何变化？',
             '逐题效用 U ＝ Judge 总体分 − λ × 工作流 token / 48,000；只比较同一档三种策略都有有效值的题。', edition, 3), read_csv(paths[2])), paths[2:]),
    ]


def early_chart(directory, palette):
    files = [directory / 'early.csv', directory / 'early_summary.csv']
    data = keyed(read_csv(files[0]), 'config', 'step')
    summaries = keyed(read_csv(files[1]), 'config', 'seed')
    steps, seeds = list(range(100, 1501, 100)), ('7', '17', '27')
    groups = ('base', 'buffer_small', 'buffer_large', 'target_frequent', 'target_rare')
    require(set(data) == {(g, str(s)) for g in groups for s in steps} and
            set(summaries) == {(g, seed) for g in groups for seed in seeds}, '需要五组×15 个检查点×三个种子')
    for row in data.values():
        vals = [number(row['seed'+s]) for s in seeds]
        require(None not in vals and all(math.isclose(number(row[k]), v, abs_tol=1e-12) for k, v in
                [('min_return', min(vals)), ('max_return', max(vals)), ('mean_return', fmean(vals))]), '种子带与均值不一致')
    vals = [number(r[k]) for r in data.values() for k in ('min_return', 'max_return')]
    ticks = nice_axis(min([0, *vals]), max([1, *vals]), 4)
    bottom = max(ticks[0], min([0, *vals]) - .07 * (ticks[-1] - ticks[0]))
    ticks = [t for t in ticks if t >= bottom]
    def at(group, s, k='mean_return'):
        return number(data[group, str(s)][k])
    sheet = Sheet(palette, 'v2 早期训练：回放池与同步间隔改变了什么？',
                  f'共享基准从第 100 步的 {at("base",100):.2f} 升至第 200 步的 {at("base",200):.2f}；频繁同步在第 300 步回落至 {at("target_frequent",300):.2f}（种子 7：{at("target_frequent",300,"seed7"):.2f}）。',
                  '训练内仿真   /   不含独立测试题及真实 Judge 分数', 4)
    configs = {g: summaries[g, seeds[0]] for g in groups}
    panels = (
        ('capacity', f'A   回放池容量（条）     /     固定目标网络每 {int(configs["base"]["target_interval"]):,} 次更新同步',
         (('buffer_small', '小容量', 1), ('base', '共享基准', 0), ('buffer_large', '大容量', 2))),
        ('target_interval', f'B   目标网络同步间隔（更新次数）     /     固定回放池 {int(configs["base"]["capacity"]):,} 条',
         (('target_frequent', '频繁同步', 1), ('base', '共享基准', 0), ('target_rare', '稀疏同步', 2))),
    )
    for i, (field, label, panel_groups) in enumerate(panels):
        top = 213 + 264*i
        sheet.text(56, top, label, 23, bold=True)
        if i == 0:
            sheet.text(1384, top, '纵轴：模拟验证平均回报', 18, anchor='end', secondary=True)
        for j, (group, title, slot) in enumerate(panel_groups):
            fixed = 'target_interval' if field == 'capacity' else 'capacity'
            require(all(summaries[group, seed][fixed] == configs['base'][fixed] and
                        summaries[group, seed][field] == configs[group][field] for seed in seeds), f'{group} 配置不一致')
            color = sheet.colors[slot]
            x, y, w, h = 103 + j*450, top+62, 370, 148
            X = lambda s: x + w*(s-100)/1400
            Y = lambda v: y + h*(ticks[-1]-v)/(ticks[-1]-bottom)
            key_label(sheet, x, top+40, color, f'{title}   {int(configs[group][field]):,}')
            for tick in ticks:
                sheet.line(x, Y(tick), x+w, Y(tick), sheet.grid)
                sheet.text(x-10, Y(tick)+6, f'{tick:g}', 16, anchor='end', secondary=True)
            for s in (100, 500, 1000, 1500):
                sheet.text(X(s), y+h+22, f'{s:,}', 16, anchor='middle', secondary=True)
            coords = lambda k: [(X(s), Y(at(group, s, k))) for s in steps]
            sheet.path(coords('max_return') + coords('min_return')[::-1], color, band=True)
            sheet.path(coords('mean_return'), color)
            for s in ((100, 300, 1500) if group == 'target_frequent' else (100, 1500)):
                sheet.dot(X(s), Y(at(group, s)), color)
            for s in steps:
                hover_box(sheet, X(s)-w/28, y, w/14, h, f'{title} · 第 {s} 步 · 均值 {at(group,s):.3f}；种子范围 {at(group,s,"min_return"):.3f} 至 {at(group,s,"max_return"):.3f}')
    sheet.text(56, 732, '横轴：环境步数，100–1,500。线为 3 个种子的均值，阴影为最小至最大值（非置信区间）；两行共用同一基准。', 18, secondary=True)
    svg = sheet.finish('数据：early/early.csv、early/early_summary.csv   ·   每 100 步一点评估；均为训练内模拟回报，不是独立十题终评。')
    return '04_v2_early_ablation', svg, files


def write_figures(figures, out, svg_only=False):
    renderer = None if svg_only else shutil.which('sips')
    require(svg_only or renderer, '未找到 macOS sips；请使用 --svg-only 输出可编辑 SVG')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'sources.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('figure_stem', 'input_csv_relative_to_figures'))
        for name, content, files in figures:
            refs = [os.path.relpath(p, out) for p in files]
            for ref in refs:
                writer.writerow([name, ref])
            meta = '<metadata>Source CSV (relative to this SVG): ' + escape('; '.join(refs)) + '</metadata>'
            svg, png = out / f'{name}.svg', out / f'{name}.png'
            svg.write_text(content.replace('</svg>', meta + '</svg>'), encoding='utf-8')
            if renderer:
                subprocess.run([renderer, '-s', 'format', 'png', str(svg), '--out', str(png)],
                               check=True, stdout=subprocess.DEVNULL)
    pngs = f' + {len(figures)} PNG (1920×1080)' if renderer else ''
    print(f'{out}: {len(figures)} editable SVG{pngs}; source index: sources.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--early-only', action='store_true', help='只读已有 v2 训练内仿真统计表')
    p.add_argument('--theme', choices=('light', 'dark'), default='light')
    p.add_argument('--out-dir', type=Path, help='outputs/ 下的新目录，默认 outputs/course-final/figures（早期图为 early_figures）')
    p.add_argument('--svg-only', action='store_true', help='只生成 SVG 和来源索引，不需要 macOS sips')
    args = p.parse_args()
    palette = LIGHT if args.theme == 'light' else DARK
    out = new_output(args.out_dir or BATCH / ('early_figures' if args.early_only else 'figures'))
    charts = [] if args.early_only else final_charts(PUBLIC / 'report', palette)
    write_figures([*charts, early_chart(PUBLIC / 'early', palette)], out, args.svg_only)


if __name__ == '__main__':
    main()
