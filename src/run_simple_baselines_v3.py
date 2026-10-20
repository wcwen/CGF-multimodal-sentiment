# -*- coding: utf-8 -*-
"""Text-only 与 Concat 两个简单基线（各 4 种子，V3 协议）。

目的（审稿意见 6）：为"增益从哪来"提供参照系——
  - TextOnly：只吃池化文本向量，检验文本单模态强度（MSA 中文本主导的下限参照）；
  - Concat：池化三模态向量直接拼接 + 同构 MLP，检验"简单融合"基线。
协议与 V3 主表完全一致（冻结权重、balanced=2.0、focal γ=3 + ls 0.05、
种子 {42,2026,7,3407}、训满 40 epoch 取末轮、hidden 256、dropout 0.1）。

结果：progress_simple_baselines_v3.json（键 模型/种子），
权重 saved_models/v3_TextOnly_{seed}.pt、v3_Concat_{seed}.pt。
断点续跑：重复执行直至完成。
用法: python run_simple_baselines_v3.py [时间预算秒，默认240]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sucai', 'code'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch import nn

from common import (load_mosi, make_loaders, set_seed, train_one_run, evaluate,
                    compute_class_weights, to_ternary_labels, Progress, SEEDS)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_simple_baselines_v3.json')
PROTOCOL = 'v3-frozen-weights'


def mlp(in_dim, hidden=256, num_classes=3, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, num_classes))


class TextOnly(nn.Module):
    """只吃池化文本向量 (B,768)。"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.net = mlp(768, num_classes=num_classes)

    def forward(self, text, audio, vision):
        return self.net(text)


class Concat(nn.Module):
    """池化三模态向量拼接 (B,793) + MLP。"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.net = mlp(768 + 5 + 20, num_classes=num_classes)

    def forward(self, text, audio, vision):
        return self.net(torch.cat([text, audio, vision], dim=-1))


MODELS = {'TextOnly': TextOnly, 'Concat': Concat}


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
            'select_metric': 'last', 'gamma': 3.0, 'label_smoothing': 0.05}

    prog = Progress(OUT)
    t0 = time.time()
    total = len(MODELS) * len(SEEDS)
    finished = True

    for name, cls in MODELS.items():
        for seed in SEEDS:
            key = f'{name}/{seed}'
            if prog.has(key):
                continue
            if time.time() - t0 > BUDGET:
                finished = False
                break
            set_seed(seed)
            loaders = make_loaders(data, 'ternary', pooled=True, seed=seed,
                                   balanced=2.0)
            model = cls(num_classes=3)
            print(f'[{time.time()-t0:.0f}s] {name}/seed{seed} ...', flush=True)
            ckpt = os.path.join(HERE, 'tmp_ckpt', f'v3{name}_{seed}.pt')
            model = train_one_run(model, loaders[0], loaders[1], is_cgf=False,
                                  resume_path=ckpt, select_metric='last',
                                  class_weights=w.clone())
            torch.save(model.state_dict(), os.path.join(
                HERE, 'saved_models', f'v3_{name}_{seed}.pt'))
            r = evaluate(model, loaders[2], dump=False)
            entry = dict(meta)
            entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                          'f1_per_class': r['f1_per_class'],
                          'labels': r['labels'], 'preds': r['preds'],
                          'train_seed': seed, 'model': name,
                          'input_format': 'pooled vectors'})
            prog.set(key, entry)
            if os.path.exists(ckpt):
                os.remove(ckpt)
            print(f'  -> acc={r["accuracy"]*100:.2f}%  mf1={r["macro_f1"]:.4f}',
                  flush=True)

    print(f'\n已完成 {len(prog.data)}/{total};',
          '全部完成' if finished else '请再次运行本脚本继续')


if __name__ == '__main__':
    main()
