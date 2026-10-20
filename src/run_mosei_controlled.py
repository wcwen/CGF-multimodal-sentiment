# -*- coding: utf-8 -*-
"""MOSEI 全量受控对照：CGF-full vs Constant-c，4 训练种子。

目的：在第二个数据集上按与 MOSI V3 完全一致的协议重复受控对照，
回答"样本级置信度是否带来可测收益"，并重复置信度诊断分析。

协议（与 MOSI V3 / 试点相同）：
  冻结类别权重（原始标签公式 ((N+s)/(Nk+s))^0.7，s=0.01N，截断[0.5,15]，
  归一化均值 1）；稀有类 2 倍过采样；focal γ=3 + label smoothing 0.05；
  λ=0.5；训满 40 epochs 取末轮（无早停、无验证集选择）；
  AdamW 2e-4 / weight decay 1e-4 / cosine / 梯度裁剪 1.0；batch 32；d=256, h=4。
  训练种子 {42, 2026, 7, 3407}；三件套种子同步固定。

每个运行额外记录置信度诊断（干净测试集）：
  按置信度三分位分组准确率、置信度与预测正确性的 Pearson r（逐样本）。

断点续跑：progress_mosei_controlled.json，键 variant/seed。
用法: python run_mosei_controlled.py [时间预算秒，默认6.5小时]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import (load_dataset, make_loaders, set_seed, train_one_run,
                    evaluate, compute_class_weights, to_ternary_labels, Progress)
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 23400.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_mosei_controlled.json')
CONSTANT_C = 0.5
VARIANTS = ['full', 'constantc']
SEEDS = [42, 2026, 7, 3407]
TEXT_DIM, AUDIO_DIM, VISION_DIM = 768, 74, 35


def confidence_analysis(labels, preds, conf):
    """逐样本置信度诊断：三分位分组准确率 + Pearson r(置信度, 正确性)。"""
    y = np.asarray(labels)
    p = np.asarray(preds)
    c = np.asarray(conf, dtype=np.float64)
    correct = (y == p).astype(np.float64)
    q1, q2 = np.quantile(c, [1 / 3, 2 / 3])
    terc = np.digitize(c, [q1, q2])          # 0/1/2
    terc_acc = [float(correct[terc == t].mean()) for t in (0, 1, 2)]
    terc_n = [int((terc == t).sum()) for t in (0, 1, 2)]
    r = float(np.corrcoef(c, correct)[0, 1])
    return {'tercile_acc': terc_acc, 'tercile_n': terc_n, 'pearson_r': r,
            'conf_mean': float(c.mean()), 'conf_sd': float(c.std())}


def main():
    data = load_dataset('mosei')
    y = to_ternary_labels(np.asarray(data['train']['labels']))
    counts = np.bincount(y)
    w = compute_class_weights(y)
    print(f'[MOSEI-ctrl] 训练集 {len(y)} 条，冻结类别权重（计数 {counts.tolist()}）: '
          f'{np.round(w.numpy(), 4).tolist()}', flush=True)

    meta = {'protocol': 'mosei-v1-frozen-weights', 'dataset': 'mosei', 'task': 'ternary',
            'dims': [TEXT_DIM, AUDIO_DIM, VISION_DIM],
            'class_counts': counts.tolist(),
            'class_weights': [round(float(x), 6) for x in w.numpy()],
            'balanced': 2.0, 'epochs': 40, 'select_metric': 'last',
            'gamma': 3.0, 'label_smoothing': 0.05, 'aux_weight': 0.5,
            'train_seeds': SEEDS, 'constant_c': CONSTANT_C}

    prog = Progress(OUT)
    t0 = time.time()
    total = len(VARIANTS) * len(SEEDS)
    finished = True

    for seed in SEEDS:
        set_seed(seed)
        loaders = make_loaders(data, 'ternary', pooled=False, seed=seed, balanced=2.0)
        for variant in VARIANTS:
            key = f'{variant}/{seed}'
            if prog.has(key):
                continue
            if time.time() - t0 > BUDGET:
                finished = False
                break
            if variant == 'full':
                model = CGFSeq(text_dim=TEXT_DIM, audio_dim=AUDIO_DIM,
                               vision_dim=VISION_DIM, num_classes=3)
            else:
                model = CGFSeq(text_dim=TEXT_DIM, audio_dim=AUDIO_DIM,
                               vision_dim=VISION_DIM, num_classes=3,
                               confidence_mode='constant', constant_c=CONSTANT_C)
            print(f'[{time.time()-t0:.0f}s] MOSEI {variant}/seed{seed} 开始训练 ...',
                  flush=True)
            run_t0 = time.time()
            ckpt = os.path.join(HERE, 'tmp_ckpt', f'mosei_ctrl_{variant}_{seed}.pt')
            model = train_one_run(model, loaders[0], loaders[1], is_cgf=True,
                                  resume_path=ckpt, select_metric='last',
                                  class_weights=w.clone())
            train_s = time.time() - run_t0
            torch.save(model.state_dict(), os.path.join(
                HERE, 'saved_models', f'mosei_v1_{variant}_{seed}.pt'))
            r = evaluate(model, loaders[2], dump=True)
            conf_stats = confidence_analysis(r['labels'], r['preds'], r['confidence'])
            entry = dict(meta)
            entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                          'f1_per_class': r['f1_per_class'],
                          'labels': r['labels'], 'preds': r['preds'],
                          'confidence': r['confidence'], 'conf_stats': conf_stats,
                          'train_seed': seed, 'model': f'CGF-{variant}',
                          'train_seconds': round(train_s, 1),
                          'input_format': 'sequence(50 steps)'})
            prog.set(key, entry)
            if os.path.exists(ckpt):
                os.remove(ckpt)
            print(f'  -> {variant}/seed{seed}: acc={r["accuracy"]*100:.2f}%  '
                  f'mf1={r["macro_f1"]:.4f}  r={conf_stats["pearson_r"]:.3f}  '
                  f'训练{train_s:.0f}s', flush=True)

    print(f'\n已完成 {len(prog.data)}/{total};',
          '全部完成' if finished else '请再次运行本脚本继续')

    if len(prog.data) == total:
        print('\n===== MOSEI 受控对照汇总（4 seeds）=====')
        for variant in VARIANTS:
            accs = [prog.data[f'{variant}/{s}']['accuracy'] * 100 for s in SEEDS]
            mf1s = [prog.data[f'{variant}/{s}']['macro_f1'] for s in SEEDS]
            rs = [prog.data[f'{variant}/{s}']['conf_stats']['pearson_r'] for s in SEEDS]
            ta = np.mean([prog.data[f'{variant}/{s}']['conf_stats']['tercile_acc']
                          for s in SEEDS], axis=0) * 100
            print(f'{variant:10s} acc={np.mean(accs):.2f}±{np.std(accs, ddof=1):.2f}%  '
                  f'mf1={np.mean(mf1s):.4f}±{np.std(mf1s, ddof=1):.4f}  '
                  f'r={np.mean(rs):.3f}  tercile={ta[0]:.1f}/{ta[1]:.1f}/{ta[2]:.1f}%')
            # pooled over seeds (the reporting口径 used in the paper);
            # not applicable to the constant-confidence control (zero variance)
            conf_p = np.concatenate([np.asarray(prog.data[f'{variant}/{s}']['confidence'])
                                     for s in SEEDS])
            if float(np.std(conf_p)) > 0:
                lab_p = np.concatenate([np.asarray(prog.data[f'{variant}/{s}']['labels'])
                                        for s in SEEDS])
                pred_p = np.concatenate([np.asarray(prog.data[f'{variant}/{s}']['preds'])
                                         for s in SEEDS])
                corr_p = (pred_p == lab_p).astype(float)
                qp = np.quantile(conf_p, [1 / 3, 2 / 3])
                ta_p = [corr_p[conf_p <= qp[0]].mean() * 100,
                        corr_p[(conf_p > qp[0]) & (conf_p <= qp[1])].mean() * 100,
                        corr_p[conf_p > qp[1]].mean() * 100]
                r_p = np.corrcoef(conf_p, corr_p)[0, 1]
                print(f'{variant:10s} POOLED tercile={ta_p[0]:.1f}/{ta_p[1]:.1f}/{ta_p[2]:.1f}%'
                      f'  N={len(conf_p)}  r={r_p:.3f}')
            else:
                print(f'{variant:10s} POOLED tercile n/a (constant confidence control)')


if __name__ == '__main__':
    main()
