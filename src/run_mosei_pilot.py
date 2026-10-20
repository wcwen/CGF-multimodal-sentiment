# -*- coding: utf-8 -*-
"""MOSEI 试点：Full vs Constant-c（ternary，种子 42）。

目的：测量 MOSEI 单次运行耗时（train 16,327 样本 ≈ MOSI 的 13 倍），
验证 74/35 维度与数据管线无误，为全量实验（4 种子 × 受控对照）定范围。

协议与 MOSI V3 完全一致：冻结类别权重（原始标签公式 (N+s)/(Nk+s))^0.7，
s=0.01N，截断[0.5,15]，归一化均值1；稀有类 2 倍过采样；focal γ=3 + ls 0.05；
λ=0.5；训满 40 轮取末轮；AdamW 2e-4 / cosine / 梯度裁剪 1.0。
唯一差异：输入维度 (768,74,35)，数据集 CMU-MOSEI（官方 SDK 划分）。

断点续跑：progress_mosei_pilot.json，键 variant/seed；重复执行直至完成。
用法: python run_mosei_pilot.py [时间预算秒，默认240]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from common import (load_dataset, make_loaders, set_seed, train_one_run, evaluate,
                    compute_class_weights, to_ternary_labels, Progress)

from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_mosei_pilot.json')
CONSTANT_C = 0.5
VARIANTS = ['full', 'constantc']
PILOT_SEEDS = [42]
TEXT_DIM, AUDIO_DIM, VISION_DIM = 768, 74, 35


def main():
    data = load_dataset('mosei')
    y = to_ternary_labels(np.asarray(data['train']['labels']))
    counts = np.bincount(y)
    w = compute_class_weights(y)
    print(f'[MOSEI-pilot] 训练集 {len(y)} 条，冻结类别权重（计数 {counts.tolist()}）: '
          f'{np.round(w.numpy(), 4).tolist()}', flush=True)

    meta = {'protocol': 'mosei-v1-frozen-weights', 'dataset': 'mosei', 'task': 'ternary',
            'dims': [TEXT_DIM, AUDIO_DIM, VISION_DIM],
            'class_counts': counts.tolist(),
            'class_weights': [round(float(x), 6) for x in w.numpy()],
            'balanced': 2.0, 'epochs': 40, 'select_metric': 'last',
            'gamma': 3.0, 'label_smoothing': 0.05, 'aux_weight': 0.5}

    prog = Progress(OUT)
    t0 = time.time()
    total = len(VARIANTS) * len(PILOT_SEEDS)
    finished = True

    for seed in PILOT_SEEDS:
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
            ckpt = os.path.join(HERE, 'tmp_ckpt', f'mosei_pilot_{variant}_{seed}.pt')
            model = train_one_run(model, loaders[0], loaders[1], is_cgf=True,
                                  resume_path=ckpt, select_metric='last',
                                  class_weights=w.clone())
            train_s = time.time() - run_t0
            torch.save(model.state_dict(), os.path.join(
                HERE, 'saved_models', f'mosei_pilot_{variant}_{seed}.pt'))
            ev_t0 = time.time()
            r = evaluate(model, loaders[2], dump=True)
            ev_s = time.time() - ev_t0
            entry = dict(meta)
            entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                          'f1_per_class': r['f1_per_class'],
                          'labels': r['labels'], 'preds': r['preds'],
                          'train_seed': seed, 'model': f'CGF-{variant}',
                          'train_seconds': round(train_s, 1),
                          'eval_seconds': round(ev_s, 1),
                          'input_format': 'sequence(50 steps)'})
            if variant == 'constantc':
                entry['constant_c'] = CONSTANT_C
            if 'confidence' in r:
                entry['confidence'] = r['confidence']
            if 'gate_weights' in r:
                entry['gate_weights'] = r['gate_weights']
            prog.set(key, entry)
            if os.path.exists(ckpt):
                os.remove(ckpt)
            print(f'  -> {variant}: acc={r["accuracy"]*100:.2f}%  '
                  f'mf1={r["macro_f1"]:.4f}  训练{train_s:.0f}s  评估{ev_s:.0f}s',
                  flush=True)

    print(f'\n已完成 {len(prog.data)}/{total};',
          '全部完成' if finished else '请再次运行本脚本继续')


if __name__ == '__main__':
    main()
