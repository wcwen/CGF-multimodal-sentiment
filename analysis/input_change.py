# -*- coding: utf-8 -*-
"""扰动后各模型实际输入的变化量（审稿意见 2 的量化证据）。

所有模型从同一份受损 50 步文本序列出发：
  - CGF 直接接收受损序列（sequence 输入）；
  - 基线接收受损序列的时间均值池化（pooled 输入）。
本脚本对每种扰动条件（对 3 个扰动实现取均值）量化两条输入管线实际收到的变化：
  - 相对 L2 变化 = ||x_corr - x_clean||_F / ||x_clean||_F（逐样本后取均值 ± SD）
  - 余弦相似度（逐样本展平后取均值）
并给出池化衰减比 = sequence 相对变化 / pooled 相对变化。

运行（需要 torch，用实验环境 python）：
  .venv-cgf/Scripts/python.exe analysis/input_change.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from common import load_mosi                      # noqa: E402
from run_fair_perturb_v3 import cond_instances, corrupt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, 'input_change.md')


def metrics(xc, x0):
    """xc/x0: (N, L, D) numpy。返回 (rel_l2_mean, rel_l2_sd, cos_mean)。"""
    num = np.linalg.norm((xc - x0).reshape(len(x0), -1), axis=1)
    den = np.linalg.norm(x0.reshape(len(x0), -1), axis=1).clip(min=1e-9)
    rel = num / den
    a = xc.reshape(len(x0), -1)
    b = x0.reshape(len(x0), -1)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1).clip(min=1e-9))
    return rel.mean(), rel.std(ddof=1), cos.mean()


def main():
    data = load_mosi()
    text = torch.as_tensor(data['test']['text'])          # (N, 50, 768)
    audio = torch.as_tensor(data['test']['audio'])
    vision = torch.as_tensor(data['test']['vision'])
    n = len(text)

    inst = cond_instances()
    groups = {}
    for key, spec in inst:
        if 'missing' in spec or key == 'clean':
            continue
        cond = key.split('/')[0]
        groups.setdefault(cond, []).append(spec)

    lines = ['# 扰动后实际输入变化量（CMU-MOSI 测试集，N=%d）' % n, '',
             'sequence = CGF 实际输入（受损 50 步序列）；pooled = 基线实际输入（受损序列的时间均值池化）。'
             '每个条件对 3 个扰动实现取均值。', '',
             '| 条件 | sequence 相对L2 (%) | pooled 相对L2 (%) | 池化衰减比 | sequence 余弦 | pooled 余弦 |',
             '|---|---|---|---|---|---|']
    for cond, specs in groups.items():
        rel_s, rel_p, cos_s, cos_p = [], [], [], []
        for spec in specs:
            tc, _, _ = corrupt(text, audio, vision, spec)
            tc = tc.numpy()
            m_s = metrics(tc, text.numpy())
            m_p = metrics(tc.mean(axis=1, keepdims=True), text.numpy().mean(axis=1, keepdims=True))
            rel_s.append(m_s[0]); cos_s.append(m_s[2])
            rel_p.append(m_p[0]); cos_p.append(m_p[2])
        rs, rp = np.mean(rel_s) * 100, np.mean(rel_p) * 100
        lines.append(f'| {cond} | {rs:.2f} | {rp:.2f} | {rs / rp:.2f}× | '
                     f'{np.mean(cos_s):.4f} | {np.mean(cos_p):.4f} |')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print('\n已写入', OUT_MD)


if __name__ == '__main__':
    main()
