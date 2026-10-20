# -*- coding: utf-8 -*-
"""重新评估 saved_models/v3_CGF_{seed}.pt，导出 MOSI 测试集逐样本 preds+confidence。

前置：先运行 `python src/run_v3_main.py <seed>` 生成 src/saved_models/v3_CGF_{seed}.pt
（权重文件体积大，不随仓库发布）。
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', 'src'))
sys.path.insert(0, SRC)
from common import (DEVICE, SEEDS, evaluate, load_mosi, make_loaders,  # noqa: E402
                    set_seed, to_ternary_labels)
from models_v2 import CGFSeq  # noqa: E402

OUT = os.path.join(HERE, 'mosi_test_records.json')

data = load_mosi()
y = to_ternary_labels(np.asarray(data['test']['labels']))
recs = {}
for seed in SEEDS:
    wpath = os.path.join(SRC, 'saved_models', f'v3_CGF_{seed}.pt')
    model = CGFSeq(text_dim=768, audio_dim=5, vision_dim=20, num_classes=3)
    model.load_state_dict(torch.load(wpath, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    set_seed(seed)
    loaders = make_loaders(data, 'ternary', pooled=False, seed=seed, balanced=2.0)
    r = evaluate(model, loaders[2], dump=True)
    pred = np.asarray(r['preds'], dtype=int)
    acc = (pred == y).mean()
    print(f'seed{seed}: acc={acc*100:.2f}%  n_conf={len(r["confidence"])}')
    recs[str(seed)] = {'preds': pred.tolist(),
                       'confidence': [float(c) for c in r['confidence']]}

json.dump(recs, open(OUT, 'w', encoding='utf-8'))
print('saved', OUT)
