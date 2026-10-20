# -*- coding: utf-8 -*-
"""Fig. 2 统计重提取：CGF 在文本扰动下的置信度与门控响应（V3 协议）。

扰动实例与 run_fair_perturb_v3.py 完全一致：同一份原始 50 步测试序列，
扰动由 (条件, 强度, 扰动种子) 经 crc32 固定，与训练种子无关。
对每个 (条件实例, 训练种子)：加载 saved_models/v3_CGF_{seed}.pt，
前向推理并记录每个测试样本的 confidence 与 gate_weights（dump）。

聚合（与 Table 4 嵌套口径一致）：
  条件内先对 3 个扰动实现取样本均值 -> 得到每个训练种子的 (conf, gate) 均值
  再对 4 个训练种子取 mean ± SD（样本标准差 ddof=1）。

断点续跑：progress_fig2_stats.json。用法: python run_fig2_stats.py [预算秒]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import load_mosi, to_ternary_labels, Progress, SEEDS
from run_fair_perturb_v3 import cond_instances, corrupt, build_model

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 6000.0
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'progress_fig2_stats.json')


@torch.no_grad()
def run_dump(model, text_s, audio_s, vision_s, batch=256):
    model.eval()
    confs, gates = [], []
    for i in range(0, text_s.shape[0], batch):
        out = model(text_s[i:i + batch], audio_s[i:i + batch],
                    vision_s[i:i + batch])
        confs.extend(out['confidence'].cpu().numpy().tolist())
        gates.extend(out['gate_weights'].cpu().numpy().tolist())
    c = np.asarray(confs)
    g = np.asarray(gates)  # (N, 2) acoustic/visual gate
    return {'conf_mean': float(c.mean()), 'conf_sd': float(c.std()),
            'gate_a_mean': float(g[:, 0].mean()), 'gate_v_mean': float(g[:, 1].mean())}


def main():
    data = load_mosi()
    d = data['test']
    text = torch.tensor(np.asarray(d['text']), dtype=torch.float32)
    audio = torch.tensor(np.asarray(d['audio']), dtype=torch.float32)
    vision = torch.tensor(np.asarray(d['vision']), dtype=torch.float32)

    prog = Progress(OUT)
    t0 = time.time()
    total = len(cond_instances()) * len(SEEDS)
    finished = True

    for ck, spec in cond_instances():
        if ck.startswith('missing'):
            continue  # Fig.2 只关心文本扰动
        ct, ca, cv = corrupt(text, audio, vision, spec)
        for seed in SEEDS:
            key = f'{ck}/{seed}'
            if prog.has(key):
                continue
            if time.time() - t0 > BUDGET:
                finished = False
                break
            model = build_model('CGF', seed)
            r = run_dump(model, ct, ca, cv)
            del model
            prog.set(key, r)
            print(f'[{time.time()-t0:5.0f}s] {ck:24s} seed{seed} '
                  f'conf={r["conf_mean"]:.3f} gate_a={r["gate_a_mean"]:.3f} '
                  f'gate_v={r["gate_v_mean"]:.3f}', flush=True)

    print(f'\n已完成 {len(prog.data)}/{total};',
          '全部完成' if finished else '请再次运行本脚本继续')

    # 汇总输出（嵌套口径：条件内先按扰动种子平均，再跨训练种子汇总）
    if finished:
        print('\n===== Fig.2 汇总（mean ± SD over 4 training seeds）=====')
        # 按基础条件聚合：键形如 noise_text_0.5/ps0/42
        base = {}
        for key in prog.data:
            ck, seed = key.rsplit('/', 1)          # ck='noise_text_0.5/ps0', seed='42'
            b = ck.split('/ps')[0]                  # 'noise_text_0.5'
            base.setdefault(b, {})
            base[b].setdefault(ck, {})[seed] = prog.data[key]
        for b in ['clean', 'noise_text_0.5', 'noise_text_1.0',
                  'mask_element_0.25', 'mask_element_0.5', 'mask_element_0.75',
                  'mask_step_0.25', 'mask_step_0.5', 'mask_step_0.75']:
            if b not in base:
                continue
            sub = base[b]
            # 嵌套口径：每个训练种子内先对扰动实现求均值，再跨训练种子汇总
            per_seed = []
            for s in SEEDS:
                rows = [sub[ck][str(s)] for ck in sorted(sub)
                        if str(s) in sub[ck]]
                if not rows:
                    break
                per_seed.append({k: float(np.mean([r[k] for r in rows]))
                                 for k in rows[0]})
            if len(per_seed) != len(SEEDS):
                continue
            line = f'{b:22s}'
            for k in ['conf_mean', 'gate_a_mean', 'gate_v_mean']:
                vals = [r[k] for r in per_seed]
                line += f'  {k}={np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}'
            print(line)


if __name__ == '__main__':
    main()
