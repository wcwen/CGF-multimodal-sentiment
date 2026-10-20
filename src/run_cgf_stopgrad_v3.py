# -*- coding: utf-8 -*-
"""CGF-StopGrad 变体：门控/增强路径使用 detach 后的置信度（4 种子）。

目的（审稿意见 3 的机制对照）：完整模型中主任务损失可经门控与置信度
反向传播到文本分类器，置信度分数因此可能同时反映"文本预测分布的集中程度"
与"主任务对门控的优化需求"。本变体把注入增强/门控的置信度 detach，
文本分类头仅受辅助文本分类任务监督——若置信度–正确性关联在该变体中
依然存在，则说明关联来自文本任务本身，而非主任务梯度耦合。

协议与 V3 主表完全一致（冻结权重、balanced=2.0、focal、λ=0.5、
种子 {42,2026,7,3407}、训满 40 epoch 取末轮）。

结果：progress_cgf_stopgrad_v3.json（键 CGF-StopGrad/种子），
权重 saved_models/v3stopgrad_CGF_{seed}.pt。断点续跑：重复执行直至完成。
用法: python run_cgf_stopgrad_v3.py [时间预算秒，默认240]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sucai', 'code'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import (load_mosi, make_loaders, set_seed, train_one_run, evaluate,
                    compute_class_weights, to_ternary_labels, Progress, SEEDS)
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_cgf_stopgrad_v3.json')
PROTOCOL = 'v3-frozen-weights'


def main():
    data = load_mosi()
    ytr = to_ternary_labels(np.asarray(data['train']['labels']))
    counts = np.bincount(ytr)
    w = compute_class_weights(ytr)          # 与 V3 主表同一冻结权重
    print(f'[{PROTOCOL}] 冻结类别权重（计数 {counts.tolist()}）: '
          f'{np.round(w.numpy(), 4).tolist()}', flush=True)

    meta = {'protocol': PROTOCOL, 'task': 'ternary',
            'class_counts': counts.tolist(),
            'class_weights': [round(float(x), 6) for x in w.numpy()],
            'seeds': SEEDS, 'balanced': 2.0, 'epochs': 40,
            'select_metric': 'last', 'gamma': 3.0, 'label_smoothing': 0.05,
            'aux_weight': 0.5}

    prog = Progress(OUT)
    t0 = time.time()
    total = len(SEEDS)
    finished = True

    for seed in SEEDS:
        key = f'CGF-StopGrad/{seed}'
        if prog.has(key):
            continue
        if time.time() - t0 > BUDGET:
            finished = False
            break
        set_seed(seed)
        loaders = make_loaders(data, 'ternary', pooled=False, seed=seed,
                               balanced=2.0)
        model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3,
                       confidence_stop_grad=True)
        print(f'[{time.time()-t0:.0f}s] CGF-StopGrad/seed{seed} ...', flush=True)
        ckpt = os.path.join(HERE, 'tmp_ckpt', f'v3stopgrad_CGF_{seed}.pt')
        model = train_one_run(model, loaders[0], loaders[1], is_cgf=True,
                              resume_path=ckpt, select_metric='last',
                              class_weights=w.clone())
        torch.save(model.state_dict(), os.path.join(
            HERE, 'saved_models', f'v3stopgrad_CGF_{seed}.pt'))
        r = evaluate(model, loaders[2], dump=True)
        entry = dict(meta)
        entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                      'f1_per_class': r['f1_per_class'],
                      'labels': r['labels'], 'preds': r['preds'],
                      'train_seed': seed, 'model': 'CGF-StopGrad',
                      'input_format': 'sequence(50 steps)'})
        if 'confidence' in r:
            entry['confidence'] = r['confidence']
        if 'gate_weights' in r:
            entry['gate_weights'] = r['gate_weights']
        prog.set(key, entry)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        print(f'  -> acc={r["accuracy"]*100:.2f}%  mf1={r["macro_f1"]:.4f}',
              flush=True)

    print(f'\n已完成 {len(prog.data)}/{total};',
          '全部完成' if finished else '请再次运行本脚本继续')


if __name__ == '__main__':
    main()
