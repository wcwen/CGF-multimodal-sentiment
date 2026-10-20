# -*- coding: utf-8 -*-
"""V3 统一协议主表重训：CGF + 10 个基线 × 4 种子 = 44 次运行。

导师要求的"锁定项"（与 progress_controlled_pair 同一冻结权重）：
  - 类别权重：由原始训练标签按论文公式一次性计算，全部 44 次运行共用，
    并连同计数写入每条结果（可追溯、与抽样无关）；
  - 采样：中性类双倍过采样（带放回），balanced=2.0；
  - 损失：focal γ=3 + label smoothing 0.05，辅助任务权重 λ=0.5；
  - 随机种子 {42,2026,7,3407}：每颗种子先 set_seed 再建加载器再建模型；
  - 断点恢复：检查点保存 torch/numpy/python 随机状态与类别权重，
    恢复后与连续运行完全一致（epoch 边界精确续跑）。
协议其余部分与 V2 相同（训满 40 轮取末轮，AdamW/cosine/裁剪等见 common.py）。
结果：progress_ternary_v3.json（键 模型/种子），权重 saved_models/v3_*.pt。
断点续跑：重复执行直至完成。
用法: python run_v3_main.py [时间预算秒，默认240]
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
from run_comparison_experiment import (THNE, MLMI, AMTE, CMDIBS, MAF, MATF,
                                       CIEGNet, MLCMSSL, CHF, TeFNA)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_ternary_v3.json')
PROTOCOL = 'v3-frozen-weights'

BASELINES = {
    't-HNE': THNE, 'MLMI': MLMI, 'AMTE': AMTE, 'CMDIBS': CMDIBS, 'MAF': MAF,
    'MATF': MATF, 'CIEG-Net': CIEGNet, 'MLCM-SSL': MLCMSSL, 'CHF': CHF,
    'TeFNA': TeFNA,
}


def main():
    data = load_mosi()
    ytr = to_ternary_labels(np.asarray(data['train']['labels']))
    counts = np.bincount(ytr)
    w = compute_class_weights(ytr)          # 冻结：全体 44 次运行共用
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
    total = (1 + len(BASELINES)) * len(SEEDS)
    finished = True

    for name in ['CGF'] + list(BASELINES.keys()):
        for seed in SEEDS:
            key = f'{name}/{seed}'
            if prog.has(key):
                continue
            if time.time() - t0 > BUDGET:
                finished = False
                break
            set_seed(seed)
            if name == 'CGF':
                # CGF 为序列输入（交叉注意力需要 50 时间步序列）
                loaders = make_loaders(data, 'ternary', pooled=False, seed=seed,
                                       balanced=2.0)
                model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20,
                               num_classes=3)
                wtpl = 'v3_{}_{}.pt'
                is_cgf = True
            else:
                # 基线为向量输入模型（均值池化单向量），与 V2 主表协议一致
                loaders = make_loaders(data, 'ternary', pooled=True, seed=seed,
                                       balanced=2.0)
                model = BASELINES[name](text_dim=768, audio_dim=5, vision_dim=20,
                                        hidden_dim=256, num_classes=3)
                wtpl = 'v3_{}_{}.pt'
                is_cgf = False
            print(f'[{time.time()-t0:.0f}s] V3主表 {name}/seed{seed} ...', flush=True)
            ckpt = os.path.join(HERE, 'tmp_ckpt', f'v3main_{name}_{seed}.pt')
            model = train_one_run(model, loaders[0], loaders[1], is_cgf=is_cgf,
                                  resume_path=ckpt, select_metric='last',
                                  class_weights=w.clone())
            torch.save(model.state_dict(), os.path.join(
                HERE, 'saved_models', wtpl.format(name, seed)))
            r = evaluate(model, loaders[2], dump=is_cgf)
            entry = dict(meta)
            entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                          'f1_per_class': r['f1_per_class'],
                          'labels': r['labels'], 'preds': r['preds'],
                          'train_seed': seed, 'model': name,
                          'input_format': 'sequence(50 steps)' if is_cgf
                                          else 'pooled vectors'})
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
