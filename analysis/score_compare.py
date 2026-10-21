# -*- coding: utf-8 -*-
"""T5（评审意见 S2 / R1-M3 / R2-M2 / R3-M4）：置信度排序分数的增量价值对比。

在同一份逐样本预测上比较 5 种打分（推理-only，不重训）：
  combined   : 论文组合分 c = 0.5*(1-H_norm(p_t)) + 0.5*max(p_t)        （现有）
  text_pmax  : 文本分类器最大类别概率 max(p_t)                            （最朴素参照）
  text_nent  : 文本分类器负熵 1 - H_norm(p_t)
  main_pmax  : 最终多模态分类器 max(p_m)
  main_nent  : 最终多模态分类器 1 - H_norm(p_m)

指标（逐训练种子）：
  AURC       : risk–coverage 曲线下面积（越小越好；随机拒绝基线 = 全集错误率）
  risk@50/80 : 固定覆盖率下保留子集的错误率
  配对差      : 各分数 AURC − combined AURC（同种子同预测，正 = 不如组合分）
  类别保留率  : 50% 覆盖率下各类别被保留的比例（检验是否主要排除中性类）

用法：
  python analysis/score_compare.py mosi     # CMU-MOSI（权重 exp/saved_models/v3_CGF_{seed}.pt）
  python analysis/score_compare.py mosei    # CMU-MOSEI（权重 exp/saved_models/mosei_v1_full_{seed}.pt）
结果写入 analysis/score_compare_results.json（可断点续跑）。
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(WS, 'repo', 'CGF-multimodal-sentiment', 'src'))

import common  # noqa: E402
from models_v2 import CGFSeq  # noqa: E402

# 数据指向工作区 sucai（仓库内不存数据）
common.DATASETS['mosi']['path'] = os.path.join(WS, 'sucai', 'CMU-MOSI', 'aligned_50.pkl')
common.DATASETS['mosei']['path'] = os.path.join(WS, 'sucai', 'CMU-MOSEI', 'mosei_aligned_50.pkl')

SEEDS = [42, 7, 2026, 3407]
WEIGHT = {'mosi': 'v3_CGF_{seed}.pt', 'mosei': 'mosei_v1_full_{seed}.pt'}
DIMS = {'mosi': (768, 5, 20), 'mosei': (768, 74, 35)}
OUT = os.path.join(HERE, 'score_compare_results.json')

SCORES = ['combined', 'text_pmax', 'text_nent', 'main_pmax', 'main_nent']


@torch.no_grad()
def infer_scores(model, loader):
    """逐样本导出 5 种分数 + 预测 + 标签。"""
    model.eval()
    rec = {k: [] for k in SCORES}
    preds, labels = [], []
    for text, audio, vision, y in loader:
        text, audio, vision = text.to(common.DEVICE), audio.to(common.DEVICE), vision.to(common.DEVICE)
        out = model(text, audio, vision)
        pm = torch.softmax(out['main_logits'], dim=-1)
        pt = torch.softmax(out['text_logits'], dim=-1)
        C = pm.size(-1)
        eps = 1e-8
        h_m = -(pm * torch.log(pm + eps)).sum(-1) / np.log(C)
        h_t = -(pt * torch.log(pt + eps)).sum(-1) / np.log(C)
        rec['combined'].extend(out['confidence'].cpu().numpy().tolist())
        rec['text_pmax'].extend(pt.max(-1).values.cpu().numpy().tolist())
        rec['text_nent'].extend((1 - h_t).cpu().numpy().tolist())
        rec['main_pmax'].extend(pm.max(-1).values.cpu().numpy().tolist())
        rec['main_nent'].extend((1 - h_m).cpu().numpy().tolist())
        preds.extend(out['main_logits'].argmax(-1).cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    rec['pred'] = np.asarray(preds)
    rec['y'] = np.asarray(labels)
    return rec


# ---------- 指标 ----------
def risk_coverage(correct, score):
    """按分数降序保留。返回 coverage 网格与 risk。"""
    order = np.argsort(-np.asarray(score))
    c = np.asarray(correct)[order].astype(float)
    n = len(c)
    cum_acc = np.cumsum(c) / np.arange(1, n + 1)
    return np.arange(1, n + 1) / n, 1.0 - cum_acc


def aurc(correct, score):
    cov, risk = risk_coverage(correct, score)
    return float(np.trapezoid(risk, cov))


def risk_at(correct, score, cov_target):
    order = np.argsort(-np.asarray(score))
    c = np.asarray(correct)[order].astype(float)
    k = max(1, int(round(cov_target * len(c))))
    return float(1.0 - c[:k].mean())


def class_retention(y, correct, score, cov_target):
    """50% 覆盖率下各类别保留比例 + 保留子集准确率。"""
    order = np.argsort(-np.asarray(score))
    k = max(1, int(round(cov_target * len(order))))
    keep = order[:k]
    out = {}
    for cls in np.unique(y):
        mask = y == cls
        out[f'class{cls}'] = float((mask[keep]).sum() / mask.sum())
    out['kept_acc'] = float(np.asarray(correct)[keep].mean())
    return out


def main(ds):
    data = common.load_dataset(ds)
    _, _, test_loader = common.make_loaders(data, 'ternary', pooled=False, seed=42)
    y = common.to_ternary_labels(data['test']['labels'])
    t, a, v = DIMS[ds]
    res_file = OUT
    res = json.load(open(res_file, encoding='utf-8')) if os.path.exists(res_file) else {}

    for seed in SEEDS:
        key = f'{ds}/{seed}'
        if key in res:
            print(f'[skip] {key}')
            continue
        wpath = os.path.join(WS, 'exp', 'saved_models', WEIGHT[ds].format(seed=seed))
        model = CGFSeq(text_dim=t, audio_dim=a, vision_dim=v, num_classes=3)
        model.load_state_dict(torch.load(wpath, map_location=common.DEVICE, weights_only=True))
        model.to(common.DEVICE)
        rec = infer_scores(model, test_loader)

        assert len(rec['y']) == len(y), f'{key}: {len(rec["y"])} vs {len(y)}'
        correct = (rec['pred'] == y).astype(int)
        entry = {'n': int(len(y)), 'acc': float(correct.mean()),
                 'scores': {s: aurc(correct, rec[s]) for s in SCORES},
                 'risk50': {s: risk_at(correct, rec[s], 0.5) for s in SCORES},
                 'risk80': {s: risk_at(correct, rec[s], 0.8) for s in SCORES},
                 'retain50': {s: class_retention(y, correct, rec[s], 0.5) for s in SCORES}}
        res[key] = entry
        json.dump(res, open(res_file, 'w', encoding='utf-8'), indent=1)
        print(f'[done] {key}  acc={entry["acc"]*100:.2f}%  ' +
              '  '.join(f'{s}={entry["scores"][s]*100:.2f}' for s in SCORES))

    # ---------- 汇总表 ----------
    print(f'\n===== {ds}: per-seed AURC (%) / risk@50% (%) — combined 为基准 =====')
    hdr = 'seed      ' + ''.join(f'{s:>12s}' for s in SCORES)
    print(hdr)
    for seed in SEEDS:
        e = res[f'{ds}/{seed}']
        line = f'S{seed:<8d}'
        line += ''.join(f'{e["scores"][s]*100:>8.2f}  ' for s in SCORES)
        print(line + f'   | r50: ' + ' '.join(f'{e["risk50"][s]*100:.1f}' for s in SCORES))
    comb = np.array([res[f'{ds}/{s}']['scores']['combined'] for s in SEEDS])
    rand = np.array([res[f'{ds}/{s}']['acc'] for s in SEEDS])
    print(f'\nrandom-reject AURC (=全集错误率): {((1-rand)*100).mean():.2f}%  '
          f'(combined 相对随机: {((1-rand)-comb).mean()*100:+.2f} pp)')
    print('\n配对差 (AURC_score − AURC_combined, pp; 正=不如组合分):')
    for s in SCORES[1:]:
        d = np.array([res[f'{ds}/{sd}']['scores'][s] - res[f'{ds}/{sd}']['scores']['combined']
                      for sd in SEEDS]) * 100
        print(f'  {s:10s} per-seed: ' + ' '.join(f'{x:+.2f}' for x in d) +
              f'  mean {d.mean():+.2f} ± {d.std(ddof=1):.2f}')
    print('\n50% 覆盖率类别保留率 (combined):')
    for seed in SEEDS:
        r = res[f'{ds}/{seed}']['retain50']['combined']
        print(f'  S{seed}: neg={r["class0"]:.2f} neu={r["class1"]:.2f} pos={r["class2"]:.2f} '
              f'kept_acc={r["kept_acc"]*100:.2f}%')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mosi')
