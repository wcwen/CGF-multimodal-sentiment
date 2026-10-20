# -*- coding: utf-8 -*-
"""Shuffled-c 对照（纯推理，审稿意见：样本级置信度需与打乱控制直接比较）。

目的：检验"置信度分数与**本样本**的配对"是否对预测有贡献。
做法：加载 V3 完整模型（v3_CGF_{seed}.pt，已训练、冻结），在测试集上
  1) Pass 1：前向计算每个样本自身的置信度 c_i（原始配对）；
  2) Pass 2：将全测试集的 c 按固定排列打乱（c_external 入口注入，
     其余前向计算不变），重算准确率。
每个（训练种子 × 条件）做 20 个打乱排列（排列种子与训练种子无关，
固定为 90000+r），报告 20 次打乱准确率的 mean±SD 与原始配对的差值。

条件：clean（核对值须复现 Table 3 的 CGF 列）与 noise_text_1.0
（按 Table 4 协议对 3 个扰动实例分别打乱后再平均，核对值须复现 Table 4）。

断点续跑：progress_shuffle_control.json，键为 seed/condition/realization。
用法: python run_shuffle_control.py [预算秒]
"""
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import load_mosi, to_ternary_labels, Progress, SEEDS
from models_v2 import CGFSeq

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_shuffle_control.json')
LOG = os.path.join(HERE, 'log_shuffle_control.txt')

N_SHUFFLES = 20
PERT_SEEDS = [0, 1, 2]
SHUFFLE_OFFSET = 90000          # 打乱排列的基准种子（与训练/扰动种子隔离）
CONDITIONS = ['clean', 'noise_text_1.0']


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def corrupt_text(text, sigma, ps):
    """与 run_fair_perturb_v3.corrupt 相同口径的文本加噪（实例由 ps 唯一确定）。"""
    g = torch.Generator().manual_seed(zlib.crc32(str({'noise': sigma, 'ps': ps}).encode()) & 0x7fffffff)
    std = text.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    return text + torch.randn(text.shape, generator=g) * sigma * std


@torch.no_grad()
def forward_all(model, text, audio, vision, c_vec=None, batch=256):
    """逐批前向；c_vec 为 None 时模型自算置信度，否则注入外部置信度。
    返回 (preds, confs)，confs 始终是模型自身报告的置信度。"""
    model.eval()
    preds, confs = [], []
    n = len(text)
    for i in range(0, n, batch):
        t, a, v = text[i:i + batch], audio[i:i + batch], vision[i:i + batch]
        c_ext = None if c_vec is None else torch.tensor(c_vec[i:i + batch])
        out = model(t, a, v, c_external=c_ext)
        preds.extend(out['main_logits'].argmax(dim=-1).cpu().numpy().tolist())
        if out.get('confidence') is not None:
            confs.extend(out['confidence'].cpu().numpy().tolist())
    return np.asarray(preds), np.asarray(confs)


def acc_of(preds, y):
    return float(np.mean(preds == y))


def main():
    data = load_mosi()
    d = data['test']
    text0 = torch.tensor(np.asarray(d['text']), dtype=torch.float32)
    audio = torch.tensor(np.asarray(d['audio']), dtype=torch.float32)
    vision = torch.tensor(np.asarray(d['vision']), dtype=torch.float32)
    y = np.asarray(to_ternary_labels(np.asarray(d['labels'])))
    n = len(y)
    log(f'Shuffled-c 对照：测试集 {n} 样本，训练种子 {SEEDS}，'
        f'每个条件 {N_SHUFFLES} 个打乱排列')

    prog = Progress(OUT)
    t0 = time.time()

    for seed in SEEDS:
        w = os.path.join(HERE, 'saved_models', f'v3_CGF_{seed}.pt')
        model = CGFSeq(num_classes=3)
        model.load_state_dict(torch.load(w, map_location='cpu'))

        for cond in CONDITIONS:
            ps_list = [None] if cond == 'clean' else PERT_SEEDS
            # 每个扰动实例：原始配对的置信度与准确率
            paired = {}
            for ps in ps_list:
                text_c = text0 if ps is None else corrupt_text(text0, 1.0, ps)
                preds, confs = forward_all(model, text_c, audio, vision)
                paired[ps] = {'acc': acc_of(preds, y), 'conf': confs, 'preds': preds}
            paired_acc = float(np.mean([paired[ps]['acc'] for ps in ps_list])) * 100
            log(f'seed {seed} {cond}: 原始配对 acc={paired_acc:.2f}%')

            # 打乱对照：排列打乱“该实例下的置信度向量”，注入后重算
            for r in range(N_SHUFFLES):
                key = f'{seed}/{cond}/r{r}'
                if prog.has(key):
                    continue
                if time.time() - t0 > BUDGET:
                    log('预算耗尽，退出（重跑同一命令续跑）')
                    return
                rng = np.random.default_rng(SHUFFLE_OFFSET + r)
                perm = rng.permutation(n)
                shuf_accs = []
                for ps in ps_list:
                    c_all = paired[ps]['conf']
                    c_shuf = c_all[perm]
                    text_c = text0 if ps is None else corrupt_text(text0, 1.0, ps)
                    preds, _ = forward_all(model, text_c, audio, vision, c_vec=c_shuf)
                    shuf_accs.append(acc_of(preds, y))
                acc_r = float(np.mean(shuf_accs)) * 100
                prog.set(key, {
                    'realization': r, 'shuffle_seed': SHUFFLE_OFFSET + r,
                    'accuracy': acc_r, 'paired_accuracy': paired_acc,
                    'per_ps_accuracy': [float(a) * 100 for a in shuf_accs],
                    'protocol': 'v3-shuffle-control-inference',
                })
                log(f'  seed {seed} {cond} r{r}: shuffled acc={acc_r:.2f}% '
                    f'(Δ={acc_r - paired_acc:+.2f} pp)')

    # 汇总
    log('===== Shuffled-c 汇总 =====')
    res = {}
    for seed in SEEDS:
        for cond in CONDITIONS:
            accs = [prog.get(f'{seed}/{cond}/r{r}')['accuracy'] for r in range(N_SHUFFLES)]
            pair = prog.get(f'{seed}/{cond}/r0')['paired_accuracy']
            res[(seed, cond)] = (float(np.mean(accs)), float(np.std(accs, ddof=1)), pair)
            log(f'seed {seed} {cond}: paired={pair:.2f}%  '
                f'shuffled={np.mean(accs):.2f}±{np.std(accs, ddof=1):.2f}%  '
                f'Δ={np.mean(accs) - pair:+.2f} pp')
    overall = {}
    for cond in CONDITIONS:
        deltas = [res[(s, cond)][0] - res[(s, cond)][2] for s in SEEDS]
        overall[cond] = float(np.mean(deltas))
        log(f'{cond}: 四种子平均 Δ(shuffled - paired) = {np.mean(deltas):+.3f} pp')


if __name__ == '__main__':
    main()
