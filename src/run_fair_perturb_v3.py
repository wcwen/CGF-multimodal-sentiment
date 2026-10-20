# -*- coding: utf-8 -*-
"""统一扰动公平重评（对应审阅意见 M1）。

核心协议：所有模型从【同一份原始 50 步文本序列】出发；
扰动由 (条件, 强度, 扰动种子) 唯一确定，与模型、训练种子无关：
  - 噪声：在原始序列上加零均值高斯噪声，std = σ × 该样本原始序列的标准差
  - 元素掩码：逐特征维 (B,L,D) 清零
  - 时间步掩码：整时间步 (B,L,1) 清零（两种掩码分别报告，不混称）
CGF 直接吃受损序列；基线吃【同一份受损序列的均值池化】——
池化发生在受损之后，保证所有模型看到相同的受损信息。
训练种子 {42,2026,7,3407} 与扰动种子 {0,1,2} 分开记录。

只推理：加载 exp/saved_models 里已训练的权重，无需任何训练。
断点续跑：进度存 progress_fair_perturb.json，重复执行直至完成。
用法: python run_fair_perturb.py [时间预算秒，默认10天]
"""
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sucai', 'code'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from common import load_mosi, to_ternary_labels, Progress, SEEDS
from models_v2 import CGFSeq
from run_comparison_experiment import (THNE, MLMI, AMTE, CMDIBS, MAF, MATF,
                                       CIEGNet, MLCMSSL, CHF, TeFNA)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 864000.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_fair_perturb_v3.json')
LOG = os.path.join(HERE, 'log_fair_perturb_v3.txt')

BASELINES = {
    't-HNE': THNE, 'MLMI': MLMI, 'AMTE': AMTE, 'CMDIBS': CMDIBS, 'MAF': MAF,
    'MATF': MATF, 'CIEG-Net': CIEGNet, 'MLCM-SSL': MLCMSSL, 'CHF': CHF,
    'TeFNA': TeFNA,
}
PERT_SEEDS = [0, 1, 2]


def cond_instances():
    """生成 (实例键, 扰动参数字典) 列表。键只含条件与扰动种子，不含模型/训练种子。"""
    inst = [('clean', {})]
    for s in (0.5, 1.0):
        for ps in PERT_SEEDS:
            inst.append((f'noise_text_{s}/ps{ps}', {'noise': s, 'ps': ps}))
    for p in (0.25, 0.5, 0.75):
        for ps in PERT_SEEDS:
            inst.append((f'mask_element_{p}/ps{ps}', {'mask_element': p, 'ps': ps}))
    for p in (0.25, 0.5, 0.75):
        for ps in PERT_SEEDS:
            inst.append((f'mask_step_{p}/ps{ps}', {'mask_step': p, 'ps': ps}))
    inst.append(('missing_audio', {'missing': 'audio'}))
    inst.append(('missing_vision', {'missing': 'vision'}))
    return inst


def corrupt(text, audio, vision, spec):
    """在原始 50 步序列上加扰动。text:(N,50,768) 等，均为 float tensor。"""
    text, audio, vision = text.clone(), audio.clone(), vision.clone()
    if 'missing' in spec:
        if spec['missing'] == 'audio':
            audio = torch.zeros_like(audio)
        else:
            vision = torch.zeros_like(vision)
    ps = spec.get('ps')
    if ps is not None:
        g = torch.Generator().manual_seed(zlib.crc32(str(spec).encode()) & 0x7fffffff)
    if 'noise' in spec:
        s = spec['noise']
        std = text.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        text = text + torch.randn(text.shape, generator=g) * s * std
    if 'mask_element' in spec:
        p = spec['mask_element']
        mask = (torch.rand(text.shape, generator=g) > p).float()
        text = text * mask
    if 'mask_step' in spec:
        p = spec['mask_step']
        mask = (torch.rand(text.shape[0], text.shape[1], 1, generator=g) > p).float()
        text = text * mask
    return text, audio, vision


@torch.no_grad()
def run_model(model, text_s, audio_s, vision_s, y, pooled, batch=256):
    """在（可能已受损的）原始序列上评估。pooled=True 的模型先对受损序列均值池化。"""
    model.eval()
    if pooled:
        text_s = text_s.mean(dim=1)
        audio_s = audio_s.mean(dim=1)
        vision_s = vision_s.mean(dim=1)
    preds = []
    for i in range(0, len(y), batch):
        out = model(text_s[i:i + batch], audio_s[i:i + batch],
                    vision_s[i:i + batch])
        logits = out['main_logits'] if isinstance(out, dict) else out
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    acc = float(np.mean(np.array(preds) == y))
    # macro-F1
    from sklearn.metrics import f1_score
    mf1 = float(f1_score(y, preds, average='macro'))
    return {'accuracy': acc, 'macro_f1': mf1}


def build_model(name, seed):
    if name == 'CGF':
        m = CGFSeq(num_classes=3)
        w = os.path.join(HERE, 'saved_models', f'v3_CGF_{seed}.pt')
    else:
        m = BASELINES[name](text_dim=768, audio_dim=5, vision_dim=20,
                            hidden_dim=256, num_classes=3)
        w = os.path.join(HERE, 'saved_models', f'v3_{name}_{seed}.pt')
    m.load_state_dict(torch.load(w, map_location='cpu'))
    return m


def main():
    data = load_mosi()
    d = data['test']
    text = torch.tensor(np.asarray(d['text']), dtype=torch.float32)
    audio = torch.tensor(np.asarray(d['audio']), dtype=torch.float32)
    vision = torch.tensor(np.asarray(d['vision']), dtype=torch.float32)
    y = to_ternary_labels(np.asarray(d['labels'])).tolist()

    prog = Progress(OUT)
    log = open(LOG, 'a', encoding='utf-8')
    t0 = time.time()
    models = ['CGF'] + list(BASELINES.keys())
    total = len(cond_instances()) * len(models) * len(SEEDS)
    done0 = len(prog.data)

    def logp(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    logp(f'\n===== 统一扰动公平重评 开始 {time.strftime("%H:%M:%S")} '
         f'（已完成 {done0}/{total}）=====')

    finished = True
    for ck, spec in cond_instances():
        ct, ca, cv = corrupt(text, audio, vision, spec)
        for name in models:
            pooled = (name != 'CGF')
            for seed in SEEDS:
                key = f'{ck}/{name}/{seed}'
                if prog.has(key):
                    continue
                if time.time() - t0 > BUDGET:
                    finished = False
                    break
                model = build_model(name, seed)
                r = run_model(model, ct, ca, cv, y, pooled)
                del model
                prog.set(key, r)
                logp(f'[{time.time()-t0:6.0f}s] {ck:26s} {name:10s} seed{str(seed):5s} '
                     f'-> acc={r["accuracy"]*100:.2f}%  mf1={r["macro_f1"]:.4f}')
        if not finished:
            break

    logp(f'===== 结束 {time.strftime("%H:%M:%S")}  已完成 {len(prog.data)}/{total}'
         f'{"，全部完成" if finished else "，时间预算用完"} =====')
    log.close()


if __name__ == '__main__':
    main()
