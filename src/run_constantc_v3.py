# -*- coding: utf-8 -*-
"""MOSI Constant-c 受控对照补齐：V3 协议，训练种子 2026。

背景：Constant-c 受控对照的 {42, 7, 3407} 三个种子与 V3 全量重训同批完成
（换种子前同一批运行，权重冻结协议一致），仅缺种子 2026（替换原 2021）。
本脚本按 V3 协议补训 constantc/2026，与 progress_controlled_pair.json 中
已有三个种子的 constantc 组成完整的 4 种子配对对照。

协议与 run_v3_main / 试点相同：冻结类别权重、稀有类 2 倍过采样、
训满 40 epochs 取末轮、focal γ=3 + ls 0.05、λ=0.5、AdamW 2e-4。

断点续跑：progress_constantc_v3.json。用法: python run_constantc_v3.py [预算秒]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np

from common import (load_dataset, make_loaders, set_seed, train_one_run,
                    evaluate, compute_class_weights, to_ternary_labels, Progress)
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_constantc_v3.json')
CONSTANT_C = 0.5
SEED = 2026


def main():
    data = load_dataset('mosi')
    y = to_ternary_labels(np.asarray(data['train']['labels']))
    w = compute_class_weights(y)
    print(f'[constantc-v3] MOSI 种子 {SEED}，冻结类别权重: '
          f'{np.round(w.numpy(), 4).tolist()}', flush=True)

    prog = Progress(OUT)
    t0 = time.time()
    key = f'constantc/{SEED}'
    if prog.has(key):
        print('已完成，直接输出结果。')
    else:
        set_seed(SEED)
        loaders = make_loaders(data, 'ternary', pooled=False, seed=SEED, balanced=2.0)
        model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3,
                       confidence_mode='constant', constant_c=CONSTANT_C)
        ckpt = os.path.join(HERE, 'tmp_ckpt', f'constantc_v3_{SEED}.pt')
        model = train_one_run(model, loaders[0], loaders[1], is_cgf=True,
                              resume_path=ckpt, select_metric='last',
                              class_weights=w.clone())
        r = evaluate(model, loaders[2], dump=False)
        torch.save(model.state_dict(), os.path.join(
            HERE, 'saved_models', f'v3_constantc_{SEED}.pt'))
        if os.path.exists(ckpt):
            os.remove(ckpt)
        prog.set(key, {
            'protocol': 'v3-frozen-weights', 'dataset': 'mosi', 'task': 'ternary',
            'class_weights': [round(float(x), 6) for x in w.numpy()],
            'balanced': 2.0, 'epochs': 40, 'select_metric': 'last',
            'gamma': 3.0, 'label_smoothing': 0.05, 'aux_weight': 0.5,
            'constant_c': CONSTANT_C, 'train_seed': SEED,
            'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
            'f1_per_class': r['f1_per_class'], 'labels': r['labels'],
            'preds': r['preds']})
        print(f'  -> constantc/seed{SEED}: acc={r["accuracy"]*100:.2f}%  '
              f'mf1={r["macro_f1"]:.4f}', flush=True)

    e = prog.data[key]
    print(f'\nconstantc/{SEED}: acc={e["accuracy"]*100:.2f}%  mf1={e["macro_f1"]:.4f}')
    # 与 full（V3 同种子）配对
    ab = np.nan
    print('（配对参考：V3 full 同种子见 progress_ablation_v3.json）')


if __name__ == '__main__':
    main()
