# -*- coding: utf-8 -*-
"""V3 主表（二分类）：CGF(序列版) + 10 个重实现基线 × 4 种子 = 44 runs。
与 V3 三分类完全同一协议：
  - 冻结类别权重：由原始训练标签（不过采样）一次性计算，全体 44 次运行共用；
  - minority-class 2 倍过采样（balanced=2.0）；
  - 输入口径：CGF 为 50 步序列（pooled=False），基线为均值池化向量（pooled=True）；
  - 每种子先 set_seed 再建 loaders 再建模型；检查点保存 RNG 状态与冻结权重。
断点续跑：进度存 progress_binary_v3.json，重复执行本脚本直至全部完成。
用法: python run_v3_binary.py [时间预算秒，默认240]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sucai', 'code'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from common import (load_mosi, make_loaders, set_seed, train_one_run, evaluate,
                    compute_class_weights, to_binary_labels, Progress, SEEDS)
from models_v2 import CGFSeq
from run_comparison_experiment import (THNE, MLMI, AMTE, CMDIBS, MAF, MATF,
                                       CIEGNet, MLCMSSL, CHF, TeFNA)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_binary_v3.json')
PROTOCOL = 'v3-frozen-weights'

BASELINES = {
    't-HNE': THNE, 'MLMI': MLMI, 'AMTE': AMTE, 'CMDIBS': CMDIBS, 'MAF': MAF,
    'MATF': MATF, 'CIEG-Net': CIEGNet, 'MLCM-SSL': MLCMSSL, 'CHF': CHF,
    'TeFNA': TeFNA,
}


def main():
    data = load_mosi()
    ytr = to_binary_labels(np.asarray(data['train']['labels']))
    counts = np.bincount(ytr)
    w = compute_class_weights(ytr)          # 冻结：全体 44 次运行共用
    print(f'[{PROTOCOL}] 冻结类别权重（计数 {counts.tolist()}）: '
          f'{np.round(w.numpy(), 4).tolist()}', flush=True)

    meta = {'protocol': PROTOCOL, 'task': 'binary',
            'class_counts': counts.tolist(),
            'class_weights': [round(float(x), 6) for x in w.numpy()],
            'seeds': SEEDS, 'balanced': 2.0, 'epochs': 40,
            'select_metric': 'last', 'gamma': 3.0, 'label_smoothing': 0.05,
            'aux_weight': 0.5,
            'label_rule': 'labels >= 0 mapped to positive (1), else negative (0)'}

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
                loaders = make_loaders(data, 'binary', pooled=False, seed=seed,
                                       balanced=2.0)
                model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20,
                               num_classes=2)
                is_cgf = True
            else:
                # 基线为向量输入模型（均值池化单向量），与 V2 主表协议一致
                loaders = make_loaders(data, 'binary', pooled=True, seed=seed,
                                       balanced=2.0)
                model = BASELINES[name](text_dim=768, audio_dim=5, vision_dim=20,
                                        hidden_dim=256, num_classes=2)
                is_cgf = False
            print(f'[{time.time()-t0:.0f}s] V3二分类 {name}/seed{seed} ...', flush=True)
            ckpt = os.path.join(HERE, 'tmp_ckpt', f'v3bin_{name}_{seed}.pt')
            model = train_one_run(model, loaders[0], loaders[1], is_cgf=is_cgf,
                                  resume_path=ckpt, select_metric='last',
                                  class_weights=w.clone())
            torch.save(model.state_dict(), os.path.join(
                HERE, 'saved_models', 'v3bin_{}_{}.pt'.format(name, seed)))
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
