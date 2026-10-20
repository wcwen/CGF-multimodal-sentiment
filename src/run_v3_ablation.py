# -*- coding: utf-8 -*-
"""V3 统一协议消融重训：5 个模块变体 + 5 个超参设置 × 4 种子 = 40 次运行。

与 run_v3_main.py 完全同一协议（冻结类别权重、采样、损失、种子、恢复语义），
保证 Table 3 主表与 Table 5/6 消融可直接在同一协议下解释。
结果：progress_ablation_v3.json（键 组/ternary/变体/种子序号），
权重 saved_models/v3_abl_*.pt。
断点续跑：重复执行直至完成。
用法: python run_v3_ablation.py [时间预算秒，默认240]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import (load_mosi, make_loaders, set_seed, train_one_run, evaluate,
                    compute_class_weights, to_ternary_labels, Progress, SEEDS)
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_ablation_v3.json')
PROTOCOL = 'v3-frozen-weights'

VARIANTS = {
    'full': dict(kw={}, aux=True),
    'no_cross_attn': dict(kw=dict(use_cross_attn=False), aux=True),
    'no_confidence': dict(kw=dict(use_confidence=False), aux=True),
    'no_gate': dict(kw=dict(use_gate=False), aux=True),
    'no_multitask': dict(kw={}, aux=False),
}
HYPER = {
    'heads2': dict(kw=dict(num_heads=2), aux=True),
    'heads8': dict(kw=dict(num_heads=8), aux=True),
    'dim64': dict(kw=dict(hidden_dim=64), aux=True),
    'dim128': dict(kw=dict(hidden_dim=128), aux=True),
    'dim512': dict(kw=dict(hidden_dim=512), aux=True),
}


def iter_jobs():
    for group, table in (('module', VARIANTS), ('hyper', HYPER)):
        for vname, spec in table.items():
            for si, seed in enumerate(SEEDS):
                yield f'{group}/ternary/{vname}/{si}', spec, seed


def main():
    data = load_mosi()
    ytr = to_ternary_labels(np.asarray(data['train']['labels']))
    counts = np.bincount(ytr)
    w = compute_class_weights(ytr)
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
    finished = True

    for key, spec, seed in iter_jobs():
        if prog.has(key):
            continue
        if time.time() - t0 > BUDGET:
            finished = False
            break
        set_seed(seed)
        loaders = make_loaders(data, 'ternary', pooled=False, seed=seed,
                               balanced=2.0)
        model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3,
                       **spec['kw'])
        print(f'[{time.time()-t0:.0f}s] V3消融 {key} ...', flush=True)
        ckpt = os.path.join(HERE, 'tmp_ckpt', 'v3abl_' + key.replace('/', '_') + '.pt')
        model = train_one_run(model, loaders[0], loaders[1],
                              is_cgf=spec['aux'], resume_path=ckpt,
                              select_metric='last', class_weights=w.clone())
        torch.save(model.state_dict(), os.path.join(
            HERE, 'saved_models', f'v3_abl_{key.replace("/", "_")}.pt'))
        r = evaluate(model, loaders[2])
        entry = dict(meta)
        entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                      'f1_per_class': r['f1_per_class'],
                      'labels': r['labels'], 'preds': r['preds'],
                      'train_seed': seed, 'variant': key})
        prog.set(key, entry)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        print(f'  -> acc={r["accuracy"]*100:.2f}%  mf1={r["macro_f1"]:.4f}',
              flush=True)

    print(f'\n已完成 {len(prog.data)}/40;',
          '全部完成' if finished else '请再次运行本脚本继续')


if __name__ == '__main__':
    main()
