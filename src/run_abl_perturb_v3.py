# -*- coding: utf-8 -*-
"""消融变体 × 统一扰动评估（4.4/4.5 联动分析）。

目的：将 CGF 的鲁棒性优势正面归因到具体组件。
协议与 run_fair_perturb_v3 完全一致：所有变体从同一份原始 50 步文本序列出发，
扰动由 (条件, 强度, 扰动种子) 唯一确定；所有变体均为序列输入（CGFSeq 家族），
直接吃受损序列，无需池化。
只推理：加载 saved_models 里 v3_abl_* 权重，无需训练。
断点续跑：进度存 progress_abl_perturb_v3.json。
用法: python run_abl_perturb_v3.py [时间预算秒，默认864000]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sucai', 'code'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from common import load_mosi, to_ternary_labels, Progress, SEEDS
from models_v2 import CGFSeq
from run_fair_perturb_v3 import cond_instances, corrupt

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 864000.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_abl_perturb_v3.json')
LOG = os.path.join(HERE, 'log_abl_perturb_v3.txt')

# 与 run_v3_ablation.py 中 VARIANTS/HYPER 完全相同的构造参数
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


def build_model(group, var, si):
    spec = (VARIANTS if group == 'module' else HYPER)[var]
    m = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3,
               **spec['kw'])
    w = os.path.join(HERE, 'saved_models',
                     f'v3_abl_{group}_ternary_{var}_{si}.pt')
    m.load_state_dict(torch.load(w, map_location='cpu'))
    return m


@torch.no_grad()
def run_model(model, text_s, audio_s, vision_s, y, batch=256):
    model.eval()
    preds = []
    for i in range(0, len(y), batch):
        out = model(text_s[i:i + batch], audio_s[i:i + batch],
                    vision_s[i:i + batch])
        logits = out['main_logits'] if isinstance(out, dict) else out
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    acc = float(np.mean(np.array(preds) == y))
    from sklearn.metrics import f1_score
    mf1 = float(f1_score(y, preds, average='macro'))
    return {'accuracy': acc, 'macro_f1': mf1}


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
    conds = cond_instances()
    jobs = [(g, v) for g, table in (('module', VARIANTS), ('hyper', HYPER))
            for v in table]
    total = len(conds) * len(jobs) * len(SEEDS)

    def logp(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    logp(f'===== 开始 {time.strftime("%H:%M:%S")}  已完成 {len(prog.data)}/{total} =====')
    for g, v in jobs:
        for si, seed in enumerate(SEEDS):
            model = build_model(g, v, si)   # 同变体同种子的各条件共用一个模型
            for cname, spec in conds:
                key = f'{g}/{v}/{cname}/si{si}'
                if prog.has(key):
                    continue
                if time.time() - t0 > BUDGET:
                    logp(f'===== 预算用完 {time.strftime("%H:%M:%S")}  '
                         f'已完成 {len(prog.data)}/{total}，请再次运行继续 =====')
                    return
                ct, ca, cv = corrupt(text, audio, vision, spec)
                r = run_model(model, ct, ca, cv, y)
                r.update({'variant': v, 'group': g, 'seed_index': si,
                          'train_seed': seed, 'condition': cname})
                prog.set(key, r)
                logp(f'[{time.time()-t0:7.0f}s] {g}/{v:15s} {cname:24s} '
                     f'si{si} -> acc={r["accuracy"]*100:.2f}%')
    logp(f'===== 结束 {time.strftime("%H:%M:%S")}  '
         f'已完成 {len(prog.data)}/{total}，全部完成 =====')


if __name__ == '__main__':
    main()
