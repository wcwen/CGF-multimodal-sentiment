# -*- coding: utf-8 -*-
"""从 V3 进度文件重建论文 Table 2-7，输出 Markdown 到 stdout。
路径相对本脚本所在目录解析，可在任意工作目录下运行。"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))

def load(f):
    for base in (HERE, RESULTS):  # fresh re-runs land in src/; paper versions in results/
        p = os.path.join(base, f)
        if os.path.exists(p):
            return json.load(open(p, encoding='utf-8'))
    raise FileNotFoundError(f + ' (neither src/ nor results/)')

BIN = load('progress_binary_v3.json')
TER = load('progress_ternary_v3.json')
ABL = load('progress_ablation_v3.json')
PERT = load('progress_fair_perturb_v3.json')
APERT = load('progress_abl_perturb_v3.json')

REF = {'TeFNA': 30, 'MLMI': 25, 't-HNE': 33, 'MAF': 34, 'MLCM-SSL': 32,
       'CIEG-Net': 11, 'AMTE': 38, 'CHF': 39, 'CMDIBS': 35, 'MATF': 31}
ORDER = ['TeFNA', 'MLMI', 't-HNE', 'MAF', 'MLCM-SSL', 'CIEG-Net', 'AMTE',
         'CHF', 'CMDIBS', 'MATF']  # 占位，实际按成绩排序


def ms(vals, ddof=1):
    """纯均值/std，不做缩放（accuracy 在收集时已×100，F1 保持小数形式）。"""
    return np.mean(vals), np.std(vals, ddof=ddof)


def by_model(d):
    out = {}
    for k, v in d.items():
        out.setdefault(v['model'], []).append(v)
    return out


def fmt(m, s, nd=2):
    return f'{m:.{nd}f}±{s:.{nd}f}'


def star_table(rows, cols_star):
    """rows: list of [name, cells...]; 对每列找最大值加*。"""
    best = {}
    for c in cols_star:
        vals = [r[1][c] for r in rows]
        mx = max(v[0] for v in vals)
        best[c] = mx
    return best


out = []

# ================= TABLE 2: 二分类 =================
bm = by_model(BIN)
rows = []
for m, runs in bm.items():
    accs = [r['accuracy'] * 100 for r in runs]
    mf1s = [r['macro_f1'] for r in runs]
    rows.append((m, ms(accs), ms(mf1s)))
rows.sort(key=lambda x: -x[1][0])
best_acc = max(r[1][0] for r in rows)
best_mf1 = max(r[2][0] for r in rows)
out.append('=== TABLE 2: Binary classification on CMU-MOSI (mean±std over 4 seeds) ===')
out.append('| Model | Accuracy | Macro-F1 |')
out.append('|---|---|---|')
for m, (ma, sa), (mm, sm) in rows:
    name = f'{m}(ours)' if m == 'CGF' else (f'{m}[{REF[m]}]' if m in REF else m)
    a = fmt(ma, sa) + ('*' if abs(ma - best_acc) < 1e-9 else '')
    f = fmt(mm, sm, 4) + ('*' if abs(mm - best_mf1) < 1e-9 else '')
    out.append(f'| {name} | {a} | {f} |')
out.append('')

# ================= TABLE 3: 三分类 =================
tm = by_model(TER)
rows = []
for m, runs in tm.items():
    accs = [r['accuracy'] * 100 for r in runs]
    mf1s = [r['macro_f1'] for r in runs]
    nf1 = [r['f1_per_class'][0] for r in runs]
    nef = [r['f1_per_class'][1] for r in runs]
    pf1 = [r['f1_per_class'][2] for r in runs]
    rows.append((m, ms(accs), ms(mf1s), ms(nf1), ms(nef), ms(pf1)))
rows.sort(key=lambda x: -x[1][0])
bests = [max(r[i][0] for r in rows) for i in range(1, 6)]
out.append('=== TABLE 3: Ternary classification on CMU-MOSI (mean±std over 4 seeds) ===')
out.append('| Model | Accuracy | Macro-F1 | Negative-F1 | Neutral-F1 | Positive-F1 |')
out.append('|---|---|---|---|---|---|')
for m, *stats in rows:
    name = f'{m}(ours)' if m == 'CGF' else (f'{m}[{REF[m]}]' if m in REF else m)
    cells = []
    for i, (mm, ss) in enumerate(stats):
        nd = 2 if i == 0 else 4
        cells.append(fmt(mm, ss, nd) + ('*' if abs(mm - bests[i]) < 1e-9 else ''))
    out.append(f"| {name} | {' | '.join(cells)} |")
out.append('')

# ================= TABLE 4: 鲁棒性（11 模型主表） =================
ptab = {}
for k, v in PERT.items():
    parts = k.split('/')
    cond, model, seed = '/'.join(parts[:-2]), parts[-2], parts[-1]
    ptab.setdefault(model, {}).setdefault(cond, {})[seed] = v['accuracy'] * 100


def nested_ms(cond_dict, clist):
    """嵌套口径：每个训练种子先平均各条件的值，再对 4 个训练种子算均值/std。"""
    per_seed = {}
    for c in clist:
        for seed, acc in cond_dict[c].items():
            per_seed.setdefault(seed, []).append(acc)
    return ms([np.mean(v) for v in per_seed.values()])


COLS = [('Clean', ['clean']),
        ('Noise σ=0.5', [f'noise_text_0.5/ps{p}' for p in (0, 1, 2)]),
        ('Noise σ=1.0', [f'noise_text_1.0/ps{p}' for p in (0, 1, 2)]),
        ('Elem p=0.25', [f'mask_element_0.25/ps{p}' for p in (0, 1, 2)]),
        ('Elem p=0.5', [f'mask_element_0.5/ps{p}' for p in (0, 1, 2)]),
        ('Elem p=0.75', [f'mask_element_0.75/ps{p}' for p in (0, 1, 2)]),
        ('Step p=0.25', [f'mask_step_0.25/ps{p}' for p in (0, 1, 2)]),
        ('Step p=0.5', [f'mask_step_0.5/ps{p}' for p in (0, 1, 2)]),
        ('Step p=0.75', [f'mask_step_0.75/ps{p}' for p in (0, 1, 2)])]
means = {}
for m in ptab:
    means[m] = [nested_ms(ptab[m], clist) for _, clist in COLS]
order4 = sorted(means, key=lambda m: -means[m][0][0])
col_bests = [max(means[m][i][0] for m in means) for i in range(len(COLS))]
out.append('=== TABLE 4: Robustness under text-feature corruption ===')
out.append('NOTE: Accuracy (%), reported as mean +/- SD over four training seeds. '
           'For stochastic corruptions, three perturbation realizations were evaluated '
           'for each training seed and averaged before computing the across-seed mean '
           'and SD. Clean and deterministic missing-modality results are computed over '
           'four training seeds. All models are evaluated on identical perturbation '
           'instances, which are fixed by (condition, strength, perturbation seed) '
           'independently of the model and training seed. * marks the highest mean in '
           'each column, with ties marked; it does not indicate statistical significance.')
out.append('| Model | ' + ' | '.join(c for c, _ in COLS) + ' |')
out.append('|---' * (len(COLS) + 1) + '|')
for m in order4:
    name = f'{m}(ours)' if m == 'CGF' else (f'{m}[{REF[m]}]' if m in REF else m)
    cells = []
    for i, (mm, ss) in enumerate(means[m]):
        cells.append(fmt(mm, ss) + ('*' if abs(mm - col_bests[i]) < 1e-9 else ''))
    out.append(f"| {name} | {' | '.join(cells)} |")
out.append('')

# ================= TABLE 5: 模块消融 =================
def abl_runs(group, var):
    out = []
    for k, v in ABL.items():
        p = k.split('/')
        if p[0] == group and p[2] == var:
            out.append(v)
    return out

VARS5 = [('CGF(full)', 'full'), ('-Attn', 'no_cross_attn'), ('-Conf', 'no_confidence'),
         ('-Gate', 'no_gate'), ('-MT', 'no_multitask')]
rows = []
for label, var in VARS5:
    runs = abl_runs('module', var)
    rows.append((label, ms([r['accuracy'] * 100 for r in runs]),
                 ms([r['macro_f1'] for r in runs])))
b5a = max(r[1][0] for r in rows)
b5f = max(r[2][0] for r in rows)
out.append('=== TABLE 5: Module ablation (ternary, mean±std over 4 seeds) ===')
out.append('| Variant | Acc(%) | Macro-F1 |')
out.append('|---|---|---|')
for label, (ma, sa), (mm, sm) in rows:
    a = fmt(ma, sa) + ('*' if abs(ma - b5a) < 1e-9 else '')
    f = fmt(mm, sm, 4) + ('*' if abs(mm - b5f) < 1e-9 else '')
    out.append(f'| {label} | {a} | {f} |')
out.append('')

# ================= TABLE 6: 超参消融 =================
HYP = [('heads=2', 'heads2'), ('heads=4 (default)', 'heads8'),  # 占位，下面修正
       ]
HYP = [('heads=2', 'heads2'), ('heads=8', 'heads8'),
       ('d=64', 'dim64'), ('d=128', 'dim128'), ('d=512', 'dim512')]
rows = []
for label, var in HYP:
    runs = abl_runs('hyper', var)
    rows.append((label, ms([r['accuracy'] * 100 for r in runs]),
                 ms([r['macro_f1'] for r in runs])))
# 加入 default（= module/full 行）
full_runs = abl_runs('module', 'full')
rows.append(('heads=4, d=256 (default)',
             ms([r['accuracy'] * 100 for r in full_runs]),
             ms([r['macro_f1'] for r in full_runs])))
b6a = max(r[1][0] for r in rows)
b6f = max(r[2][0] for r in rows)
out.append('=== TABLE 6: Hyperparameter sensitivity (ternary, mean±std over 4 seeds) ===')
out.append('| Setting | Acc(%) | Macro-F1 |')
out.append('|---|---|---|')
for label, (ma, sa), (mm, sm) in rows:
    a = fmt(ma, sa) + ('*' if abs(ma - b6a) < 1e-9 else '')
    f = fmt(mm, sm, 4) + ('*' if abs(mm - b6f) < 1e-9 else '')
    out.append(f'| {label} | {a} | {f} |')
out.append('')

# ================= TABLE 7: 变体 × 扰动家族 =================
atab = {}
for k, v in APERT.items():
    p = k.split('/')
    g, var, si = p[0], p[1], p[-1]
    cond = '/'.join(p[2:-1])
    atab.setdefault(var, {}).setdefault(cond, {})[si] = v['accuracy'] * 100


def nested7(var, clist):
    """嵌套口径：si(训练种子)内先平均，再对 4 个训练种子算均值/std。"""
    per_seed = {}
    for c in clist:
        for si, acc in atab[var][c].items():
            per_seed.setdefault(si, []).append(acc)
    return ms([np.mean(v) for v in per_seed.values()])


def fam(var, prefix):
    clist = [c for c in atab[var] if c.startswith(prefix)]
    return nested7(var, clist)


def one(var, cond):
    return nested7(var, [cond])


T7ROWS = [('Clean', lambda v: one(v, 'clean')),
          ('Noise σ=1.0', lambda v: fam(v, 'noise_text_1.0')),
          ('Elem mask p=0.5', lambda v: fam(v, 'mask_element_0.5')),
          ('Step mask p=0.25', lambda v: fam(v, 'mask_step_0.25')),
          ('Step mask p=0.5', lambda v: fam(v, 'mask_step_0.5')),
          ('Step mask p=0.75', lambda v: fam(v, 'mask_step_0.75')),
          ('Missing audio', lambda v: one(v, 'missing_audio')),
          ('Missing vision', lambda v: one(v, 'missing_vision'))]
V7 = ['full', 'no_cross_attn', 'no_confidence', 'no_gate', 'no_multitask']
L7 = ['CGF(full)', '-Attn', '-Conf', '-Gate', '-MT']
out.append('=== TABLE 7: Module variants under corruption ===')
out.append('NOTE: Same reporting protocol as Table 4: mean +/- SD over four training '
           'seeds; for stochastic corruptions, three perturbation realizations are '
           'averaged within each training seed first. All variants are evaluated on '
           'identical perturbation instances. * marks the highest mean in each column, '
           'with ties marked; it does not indicate statistical significance.')
out.append('| Condition | ' + ' | '.join(L7) + ' |')
out.append('|---' * (len(L7) + 1) + '|')
for label, fn in T7ROWS:
    cells = [fmt(*fn(v)) for v in V7]
    out.append(f'| {label} | ' + ' | '.join(cells) + ' |')
out.append('')

# ================= 置信度三分位（正文用） =================
conf_all, corr_all = [], []
for k, v in TER.items():
    if v['model'] != 'CGF':
        continue
    conf = np.asarray(v['confidence'])
    preds = np.asarray(v['preds'])
    labels = np.asarray(v['labels'])
    conf_all.append(conf)
    corr_all.append((preds == labels).astype(float))
conf = np.concatenate(conf_all)
corr = np.concatenate(corr_all)
q = np.quantile(conf, [1 / 3, 2 / 3])
lo = corr[conf <= q[0]].mean() * 100
mid = corr[(conf > q[0]) & (conf <= q[1])].mean() * 100
hi = corr[conf > q[1]].mean() * 100
r = np.corrcoef(conf, corr)[0, 1]
out.append(f'[置信度分析] 阈值 {q[0]:.4f}/{q[1]:.4f}; '
           f'低 {lo:.1f}% / 中 {mid:.1f}% / 高 {hi:.1f}%; Pearson r = {r:.3f}')

print('\n'.join(out))
