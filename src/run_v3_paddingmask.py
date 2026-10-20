# -*- coding: utf-8 -*-
"""Padding-mask 敏感性实验（审稿意见：补零位置参与注意力与池化需验证）。

与 V3 主表协议逐项一致（冻结类别权重、稀有类 2 倍过采样、训满 40 epoch
取末轮、select_metric='last'、focal γ=3 + ls 0.05、λ=0.5、AdamW 2e-4），
唯一区别：CGFSeq(use_padding_mask=True)——交叉注意力忽略填充 Key，
t_vec 与交互输出的时间平均池化只在有效步（原始输入非全零行）上求均值。

断点续跑：progress_paddingmask_v3.json，键为 CGF/{seed}。
掩码口径：text_valid = text_bert 第 2 行 attention_mask（1=真实 token 位；
MOSI 测试集中位有效长度 14/50）。交叉注意力忽略填充 Key/查询位的池化、
t_vec 与交互输出均在有效步上求均值。
用法: python run_v3_paddingmask.py [预算秒]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np

from common import (load_mosi, make_loaders, set_seed, train_one_run,
                    evaluate, compute_class_weights, to_ternary_labels,
                    Progress, SEEDS)
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_paddingmask_v3.json')
LOG = os.path.join(HERE, 'log_paddingmask_v3.txt')


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    data = load_mosi()
    y = to_ternary_labels(np.asarray(data['train']['labels']))
    w = compute_class_weights(y)
    log(f'padding-mask 敏感性：冻结类别权重 {np.round(w.numpy(), 4).tolist()}')

    prog = Progress(OUT)
    t0 = time.time()
    meta = {'protocol': 'v3-padding-mask', 'task': 'ternary',
            'dataset': 'CMU-MOSI', 'frozen_class_weights': True,
            'oversample': 2.0, 'select_metric': 'last',
            'padding_mask': 'text_bert attention_mask (true token positions)'}

    for seed in SEEDS:
        key = f'CGF/{seed}'
        if prog.has(key):
            continue
        if time.time() - t0 > BUDGET:
            log('预算耗尽，退出（重跑同一命令续跑）')
            return
        set_seed(seed)
        loaders = make_loaders(data, 'ternary', pooled=False, seed=seed,
                               balanced=2.0, with_mask=True)
        model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3,
                       use_padding_mask=True)
        log(f'CGF(padding-mask)/seed{seed} 训练中 ...')
        ckpt = os.path.join(HERE, 'tmp_ckpt', f'v3padmask_CGF_{seed}.pt')
        model = train_one_run(model, loaders[0], loaders[1], is_cgf=True,
                              resume_path=ckpt, select_metric='last',
                              class_weights=w.clone())
        torch.save(model.state_dict(), os.path.join(
            HERE, 'saved_models', f'v3padmask_CGF_{seed}.pt'))
        r = evaluate(model, loaders[2], dump=True)
        entry = dict(meta)
        entry.update({'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                      'f1_per_class': r['f1_per_class'],
                      'labels': r['labels'], 'preds': r['preds'],
                      'train_seed': seed, 'model': 'CGF(padding-mask)',
                      'input_format': 'sequence(50 steps)'})
        if 'confidence' in r:
            entry['confidence'] = r['confidence']
        if 'gate_weights' in r:
            entry['gate_weights'] = r['gate_weights']
        prog.set(key, entry)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        log(f'  seed{seed}: acc={r["accuracy"]:.2f}%  mf1={r["macro_f1"]:.4f}')

    log('===== padding-mask 汇总（对照：Table 3 CGF = 66.33±0.72%, mf1 0.5932）=====')
    accs, mf1s = [], []
    for seed in SEEDS:
        e = prog.get(f'CGF/{seed}')
        accs.append(e['accuracy'])
        mf1s.append(e['macro_f1'])
        log(f'  seed{seed}: acc={e["accuracy"]:.2f}%  mf1={e["macro_f1"]:.4f}')
    log(f'  mean±SD: acc={np.mean(accs):.2f}±{np.std(accs, ddof=1):.2f}%  '
        f'mf1={np.mean(mf1s):.4f}±{np.std(mf1s, ddof=1):.4f}')


if __name__ == '__main__':
    main()
