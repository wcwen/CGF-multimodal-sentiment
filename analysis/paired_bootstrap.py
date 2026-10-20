# -*- coding: utf-8 -*-
"""配对 bootstrap 统计（审稿意见 4：以测试样本为聚类单位的不确定性报告）。

内容：
1. CMU-MOSI 二分类/三分类：CGF 与每个基线的逐样本配对准确率差（4 训练种子平均），
   按测试样本聚类的 bootstrap（10,000 次重抽样）给出均值差的 95% 百分位置信区间。
2. 置信度三分位准确率的聚类 bootstrap 置信区间（MOSI 与 MOSEI），
   口径与论文一致：每个种子内分三分位、再对 4 种子取平均；
   另给 pooled Pearson r 的置信区间。
3. Macro-F1 无法逐样本分解，仅报告 4 种子描述性统计（不推断）。

运行：python analysis/paired_bootstrap.py   （仅依赖 numpy，结果写入 analysis/paired_bootstrap.md）
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_MD = os.path.join(HERE, 'paired_bootstrap.md')

N_BOOT = 10000
SEEDS = [42, 2026, 7, 3407]
MODELS = ['CGF', 't-HNE', 'MLMI', 'AMTE', 'CMDIBS', 'MAF', 'MATF',
          'CIEG-Net', 'MLCM-SSL', 'CHF', 'TeFNA']
RNG = np.random.default_rng(20261016)


def load(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d.get('data', d)


def per_seed_correct(data, task):
    """返回 {model: (4, N) bool 逐样本正确矩阵} 与共享标签。"""
    correct, labels = {}, None
    for m in MODELS:
        rows = []
        for s in SEEDS:
            e = data[f'{m}/{s}']
            y = np.asarray(e['labels'])
            p = np.asarray(e['preds'])
            rows.append(p == y)
            if labels is None:
                labels = y
        correct[m] = np.stack(rows)
    return correct, labels


def paired_bootstrap(cgf_mat, base_mat, n_boot=N_BOOT):
    """cgf_mat/base_mat: (4, N)。逐样本配对差（4 种子平均）→ 按样本聚类 bootstrap。
    返回 (mean_diff_pp, lo, hi, p_sign)。"""
    d = (cgf_mat.astype(float) - base_mat.astype(float)).mean(axis=0)  # (N,)
    n = len(d)
    obs = d.mean() * 100
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boots[b] = d[idx].mean() * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_sign = min((boots <= 0).mean(), (boots >= 0).mean()) * 2
    return obs, lo, hi, min(p_sign, 1.0)


def tercile_bootstrap(correct_all, conf_all, n_boot=N_BOOT):
    """correct_all/conf_all: list of 4 arrays (N,)。口径与论文一致（pooled）：
    合并 4 种子预测 → 统一三分位 → 组内准确率。
    聚类 bootstrap：重抽样样本，取该样本全部 4 种子预测后合并。返回 (tercile%, CIs, r, r CI)。"""
    n = len(correct_all[0])
    conf_p = np.concatenate(conf_all)
    corr_p = np.concatenate(correct_all).astype(float)
    conf_4 = np.stack(conf_all)      # (4, N)
    corr_4 = np.stack([c.astype(float) for c in correct_all])

    def pooled_terc(cf4, cr4):
        cf, cr = cf4.ravel(), cr4.ravel()
        q1, q2 = np.quantile(cf, [1 / 3, 2 / 3])
        return np.array([cr[cf <= q1].mean(),
                         cr[(cf > q1) & (cf <= q2)].mean(),
                         cr[cf > q2].mean()]) * 100

    terc_obs = pooled_terc(conf_4, corr_4)
    r_obs = np.corrcoef(conf_p, corr_p)[0, 1]

    boots_t = np.empty((n_boot, 3))
    boots_r = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boots_t[b] = pooled_terc(conf_4[:, idx], corr_4[:, idx])
        cp = conf_4[:, idx].ravel()
        rp = corr_4[:, idx].ravel()
        boots_r[b] = np.corrcoef(cp, rp)[0, 1]
    ci_t = np.percentile(boots_t, [2.5, 97.5], axis=0)
    ci_r = np.percentile(boots_r, [2.5, 97.5])
    return terc_obs, ci_t, r_obs, ci_r


def main():
    lines = ['# 配对 Bootstrap 统计报告', '',
             '聚类单位 = 测试样本（686 / 4,662 个）；每个样本携带其 4 个训练种子的预测。'
             '重抽样 10,000 次，报告均值差 95% 百分位置信区间。', '']

    # ---------- 1. MOSI 二分类/三分类配对比较 ----------
    for task, fname in [('binary', 'progress_binary_v3.json'),
                        ('ternary', 'progress_ternary_v3.json')]:
        data = load(os.path.join(ROOT, 'results', fname))
        correct, labels = per_seed_correct(data, task)
        n = len(labels)
        lines.append(f'## 1.{1 if task == "binary" else 2} CMU-MOSI '
                     f'{"二分类" if task == "binary" else "三分类"}（N={n} 测试样本 × 4 训练种子）')
        lines.append('')
        lines.append('| 对比 | 配对均值差 (pp) | 95% CI |  bootstrap 双侧 p | 结论 |')
        lines.append('|---|---|---|---|---|')
        acc = {m: correct[m].mean() * 100 for m in MODELS}
        for m in MODELS[1:]:
            obs, lo, hi, p = paired_bootstrap(correct['CGF'], correct[m])
            if lo > 0:
                concl = 'CGF 更高（CI 不含 0）'
            elif hi < 0:
                concl = '基线更高（CI 不含 0）'
            else:
                concl = 'CI 含 0，差异不显著'
            lines.append(f"| CGF − {m} | {obs:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.3f} | {concl} |")
        lines.append('')
        # Macro-F1 描述性
        lines.append('Macro-F1（4 种子描述性，不推断）：')
        for m in MODELS:
            mf = [data[f'{m}/{s}']['macro_f1'] for s in SEEDS]
            lines.append(f'- {m}: {np.mean(mf):.4f} ± {np.std(mf, ddof=1):.4f}')
        lines.append('')

    # ---------- 2. 置信度三分位 ----------
    lines.append('## 2. 置信度三分位准确率的聚类 bootstrap 置信区间')
    lines.append('')
    lines.append('| 数据集 | 三分位 | 准确率 (%) | 95% CI |')
    lines.append('|---|---|---|---|')
    for name, path, key in [('CMU-MOSI', 'results/progress_ternary_v3.json', 'CGF'),
                            ('CMU-MOSEI', 'results/progress_mosei_controlled.json', 'full')]:
        data = load(os.path.join(ROOT, path))
        corrs, confs = [], []
        for s in SEEDS:
            e = data[f'{key}/{s}']
            corrs.append(np.asarray(e['preds']) == np.asarray(e['labels']))
            confs.append(np.asarray(e['confidence'], dtype=float))
        terc_obs, ci_t, r_obs, ci_r = tercile_bootstrap(corrs, confs)
        for i, lab in enumerate(['低', '中', '高']):
            lines.append(f"| {name} | {lab} | {terc_obs[i]:.1f} | "
                         f"[{ci_t[0][i]:.1f}, {ci_t[1][i]:.1f}] |")
        lines.append(f'| {name} | Pearson r（pooled） | {r_obs:.3f} | '
                     f'[{ci_r[0]:.3f}, {ci_r[1]:.3f}] |')
        lines.append('')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print('\n已写入', OUT_MD)


if __name__ == '__main__':
    main()
