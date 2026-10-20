# -*- coding: utf-8 -*-
"""CGF-Pooled 变体：CGF 架构吃均值池化单向量（4 种子）。

目的（审稿意见 2 的受控对照）：把 CGF 与基线的输入口径差异从
"序列 vs 池化" 中剥离出来——CGF-Pooled 与基线吃完全相同的池化输入，
但保留 CGF 的全部结构（投影→交叉注意力(单步)→置信度→门控→融合）。
若 CGF-Pooled ≈ 基线 ≫ 或 ≪ CGF(序列)，即可定位增益/差距来自
序列输入还是置信度引导结构。

协议与 V3 主表完全一致：同一冻结类别权重、balanced=2.0、focal γ=3 +
label smoothing 0.05、λ=0.5、种子 {42,2026,7,3407}（先 set_seed 再建
加载器再建模型）、训满 40 epoch 取末轮、AdamW/cosine/裁剪。

输入处理：text/audio/vision 先各自沿时间维均值池化为 (N,1,D) 单步序列
再入模——模型代码与 CGFSeq 完全相同（交叉注意力退化为单步，数学上合法），
只改输入，不动结构。因此本变体的置信度/门控输出与逐样本 dump 均保留。

结果：progress_cgf_pooled_v3.json（键 CGF-pooled/种子），
权重 saved_models/v3pooled_CGF_{seed}.pt。断点续跑：重复执行直至完成。
用法: python run_cgf_pooled_v3.py [时间预算秒，默认240]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sucai', 'code'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from common import (load_mosi, set_seed, train_one_run, evaluate,
                    compute_class_weights, to_ternary_labels, Progress, SEEDS)
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_cgf_pooled_v3.json')
PROTOCOL = 'v3-frozen-weights'


def make_loaders_t1(data, task, batch_size=32, seed=0, balanced=False):
    """与 common.make_loaders 相同，但输入为 (N,1,D) 单步序列
    （时间均值池化保留维度），供 CGFSeq 以池化口径运行。"""
    conv = to_binary_labels_t if task == 'binary' else to_ternary_labels

    def prep(split, shuffle):
        d = data[split]
        t = torch.tensor(d['text'].mean(axis=1, keepdims=True), dtype=torch.float32)
        a = torch.tensor(d['audio'].mean(axis=1, keepdims=True), dtype=torch.float32)
        v = torch.tensor(d['vision'].mean(axis=1, keepdims=True), dtype=torch.float32)
        y = torch.tensor(conv(d['labels']), dtype=torch.long)
        g = torch.Generator().manual_seed(seed)
        if shuffle and balanced:
            counts = np.bincount(y.numpy())
            rare = int(np.argmin(counts))
            w = np.where(y.numpy() == rare, float(balanced), 1.0)
            smp = WeightedRandomSampler(weights=torch.tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
            return DataLoader(TensorDataset(t, a, v, y), batch_size=batch_size,
                              sampler=smp)
        return DataLoader(TensorDataset(t, a, v, y), batch_size=batch_size,
                          shuffle=shuffle, generator=g if shuffle else None)

    return prep('train', True), prep('valid', False), prep('test', False)


def to_binary_labels_t(labels):
    return (np.asarray(labels) >= 0).astype(int)


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
        key = f'CGF-pooled/{seed}'
        if prog.has(key):
            continue
        if time.time() - t0 > BUDGET:
            finished = False
            break
        set_seed(seed)
        loaders = make_loaders_t1(data, 'ternary', seed=seed, balanced=2.0)
        model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3)
        print(f'[{time.time()-t0:.0f}s] CGF-pooled/seed{seed} ...', flush=True)
        ckpt = os.path.join(HERE, 'tmp_ckpt', f'v3pooled_CGF_{seed}.pt')
        model = train_one_run(model, loaders[0], loaders[1], is_cgf=True,
                              resume_path=ckpt, select_metric='last',
                              class_weights=w.clone())
        torch.save(model.state_dict(), os.path.join(
            HERE, 'saved_models', f'v3pooled_CGF_{seed}.pt'))
        r = evaluate(model, loaders[2], dump=True)
        entry = dict(meta)
        entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                      'f1_per_class': r['f1_per_class'],
                      'labels': r['labels'], 'preds': r['preds'],
                      'train_seed': seed, 'model': 'CGF-pooled',
                      'input_format': 'pooled vectors as 1-step sequence'})
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
