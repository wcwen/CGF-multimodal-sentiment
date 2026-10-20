# -*- coding: utf-8 -*-
"""V7.2-B1: 风险-覆盖率（risk-coverage / selective prediction）分析。
用 MOSI 4 seeds（2,744 条）与 MOSEI 4 seeds（18,648 条）的逐样本
置信度+正确性，计算置信度排序下的 risk-coverage 曲线、AURC，
并与随机拒绝基线对比。
"""
import json
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', 'src'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
sys.path.insert(0, SRC)
from common import load_dataset, to_ternary_labels  # noqa: E402

SEEDS = [42, 7, 2026, 3407]


def risk_coverage(correct, conf):
    """按置信度降序保留；返回 coverage 网格与对应 risk（1-acc）。"""
    order = np.argsort(-conf)
    c = correct[order]
    cov_grid = np.linspace(0.05, 1.0, 20)
    risks = []
    n = len(c)
    for cov in cov_grid:
        k = max(1, int(round(cov * n)))
        risks.append(1.0 - c[:k].mean())
    return cov_grid, np.array(risks)


def aurc(correct, conf):
    """Area under risk-coverage curve（coverage 0..1 梯形积分）。"""
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    n = len(c)
    cum_acc = np.cumsum(c) / np.arange(1, n + 1)
    risk = 1.0 - cum_acc
    cov = np.arange(1, n + 1) / n
    return np.trapezoid(risk, cov)


def random_baseline_risk(correct, cov_grid, n_rep=200, seed0=0):
    """随机拒绝基线：任意 coverage 下 risk = 全集错误率（期望）。"""
    err = 1.0 - correct.mean()
    return np.full_like(cov_grid, err), err


def analyze(name, correct, conf):
    correct = np.asarray(correct).astype(int)
    conf = np.asarray(conf).astype(float)
    print(f'\n===== {name} （{len(conf)} 条）=====')
    print(f'整体准确率: {correct.mean()*100:.2f}%   平均置信度: {conf.mean():.3f}')
    cov, risk = risk_coverage(correct, conf)
    rb_risk, rb_err = random_baseline_risk(correct, cov)
    a = aurc(correct, conf)
    print(f'AURC = {a*100:.3f}%-risk    随机基线 AURC = {rb_err*100:.3f}%-risk')
    print('coverage -> risk(conf) / risk(random):')
    for cv, rk, rr in zip(cov, risk, rb_risk):
        print(f'  {cv:.2f}   {rk*100:6.2f}%   {rr*100:6.2f}%')
    # 置信度三分位的准确率（与 4.6 节口径一致）
    q = np.quantile(conf, [1/3, 2/3])
    terc = [conf <= q[0], (conf > q[0]) & (conf <= q[1]), conf > q[1]]
    print('置信度三分位准确率:', [f'{correct[t].mean()*100:.1f}% (n={t.sum()})' for t in terc])
    # 低风险段收益：coverage=50% 时
    k = len(conf) // 2
    order = np.argsort(-conf)
    top_acc = correct[order[:k]].mean()
    print(f'coverage=50% 时准确率 {top_acc*100:.2f}%（全集 {correct.mean()*100:.2f}%）')
    return {'aurc': float(a), 'random_aurc': float(rb_err),
            'cov': cov.tolist(), 'risk': (risk*100).tolist(),
            'random_risk': (rb_risk*100).tolist(),
            'acc': float(correct.mean())}


def main():
    results = {}

    # ---- MOSI ----
    data = load_dataset('mosi')
    y = to_ternary_labels(data['test']['labels'])
    recs = json.load(open(os.path.join(HERE, 'mosi_test_records.json'), encoding='utf-8'))
    confs, cors = [], []
    for s in SEEDS:
        pred = np.asarray(recs[str(s)]['preds'], dtype=int)
        confs.append(np.asarray(recs[str(s)]['confidence'], dtype=float))
        cors.append((pred == y).astype(int))
        print(f'MOSI seed{s}: acc={cors[-1].mean()*100:.2f}%')
    conf = np.concatenate(confs); cor = np.concatenate(cors)
    # Pearson r（与 4.6 节一致口径）
    r = np.corrcoef(conf, cor)[0, 1]
    print(f'MOSI pooled Pearson r(conf, correct) = {r:.3f}')
    results['MOSI'] = analyze('CMU-MOSI (4 seeds pooled)', cor, conf)
    results['MOSI']['pearson_r'] = float(r)

    # ---- MOSEI ----
    data = load_dataset('mosei')
    y = to_ternary_labels(data['test']['labels'])
    p = json.load(open(os.path.join(RESULTS, 'progress_mosei_controlled.json'), encoding='utf-8'))
    confs, cors = [], []
    for s in SEEDS:
        e = p[f'full/{s}']
        pred = np.asarray(e['preds'], dtype=int)
        assert len(pred) == len(y), f'seed{s}: {len(pred)} vs {len(y)}'
        confs.append(np.asarray(e['confidence'], dtype=float))
        cors.append((pred == y).astype(int))
        print(f'MOSEI seed{s}: acc={cors[-1].mean()*100:.2f}% (json 记录 {e["accuracy"]*100:.2f}%)')
    conf = np.concatenate(confs); cor = np.concatenate(cors)
    r = np.corrcoef(conf, cor)[0, 1]
    print(f'MOSEI pooled Pearson r(conf, correct) = {r:.3f}')
    results['MOSEI'] = analyze('CMU-MOSEI (4 seeds pooled)', cor, conf)
    results['MOSEI']['pearson_r'] = float(r)

    out = os.path.join(HERE, 'risk_coverage_results.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n已保存 {out}')


if __name__ == '__main__':
    main()
