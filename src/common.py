# -*- coding: utf-8 -*-
"""CGF v2 实验公共组件：数据、损失、训练/评估（序列保真版）

与旧版差异：
1. 数据加载不再做 np.mean(axis=1)，保留 50 时间步序列（CGF 使用），
   同时提供 pooled 版本供向量型基线模型使用；
2. 全部实验固定随机种子（4 个种子取均值±标准差）；
3. 支持测试时扰动（缺失模态 / 高斯噪声）评估；
4. 支持逐样本导出（预测、置信度、门控权重），用于案例分析与指标复算。
"""
import json
import os
import time
import copy
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATASETS = {
    'mosi': {'path': os.path.join(os.path.dirname(__file__), '..', 'sucai', 'CMU-MOSI', 'aligned_50.pkl'),
             'dims': (768, 5, 20)},
    'mosei': {'path': os.path.join(os.path.dirname(__file__), '..', 'sucai', 'CMU-MOSEI', 'mosei_aligned_50.pkl'),
              'dims': (768, 74, 35)},
}
DATA_PATH = DATASETS['mosi']['path']  # 兼容旧引用
SEEDS = [42, 2026, 7, 3407]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name='mosi'):
    """name: 'mosi' | 'mosei'。返回结构一致：text(50,768)/audio/vision 序列、labels、id。"""
    with open(DATASETS[name]['path'], 'rb') as f:
        data = pickle.load(f)
    out = {}
    for split in ['train', 'valid', 'test']:
        d = data[split]
        out[split] = {
            'text': np.asarray(d['text'], dtype=np.float32),      # (N,50,768)
            'audio': np.asarray(d['audio'], dtype=np.float32),    # (N,50,D_a)
            'vision': np.asarray(d['vision'], dtype=np.float32),  # (N,50,D_v)
            'labels': np.asarray(d['regression_labels'], dtype=np.float64),
            'id': list(d['id']),
        }
        if 'text_bert' in d:
            # (N,3,50) = [input_ids, attention_mask, token_type_ids]；
            # attention_mask 提供真实 token 长度（padding-mask 实验用）
            out[split]['text_bert'] = np.asarray(d['text_bert'])
    return out


def load_mosi():
    return load_dataset('mosi')


def to_binary_labels(labels):
    return (labels >= 0).astype(int)


def to_ternary_labels(labels):
    r = np.zeros_like(labels, dtype=int)
    r[labels < -0.5] = 0
    r[(labels >= -0.5) & (labels <= 0.5)] = 1
    r[labels > 0.5] = 2
    return r


def make_loaders(data, task, batch_size=32, pooled=False, seed=0, balanced=False,
                 with_mask=False):
    """task: 'binary' | 'ternary'; pooled: 向量输入(均值池化)或序列输入;
    balanced: 训练集类平衡采样。传浮点 M 表示“最少样本类别的采样权重为 M，其余为 1”
    （M=2.0 即中性类 2 倍过采样协议）；False 表示普通随机打乱。
    with_mask: 数据集为 5 元组 (text, audio, vision, mask, y)，mask 为
    text_bert 的 attention_mask（1=真实 token 位），供 padding-mask 实验用。"""
    conv = to_binary_labels if task == 'binary' else to_ternary_labels

    def prep(split, shuffle):
        d = data[split]
        if pooled:
            t = torch.tensor(d['text'].mean(axis=1), dtype=torch.float32)
            a = torch.tensor(d['audio'].mean(axis=1), dtype=torch.float32)
            v = torch.tensor(d['vision'].mean(axis=1), dtype=torch.float32)
        else:
            t = torch.tensor(d['text'], dtype=torch.float32)
            a = torch.tensor(d['audio'], dtype=torch.float32)
            v = torch.tensor(d['vision'], dtype=torch.float32)
        y = torch.tensor(conv(d['labels']), dtype=torch.long)
        ds = TensorDataset(t, a, v, y)
        if with_mask:
            m = torch.tensor(np.asarray(d['text_bert'])[:, 1, :],
                             dtype=torch.float32)
            ds = TensorDataset(t, a, v, m, y)
        g = torch.Generator().manual_seed(seed)
        if shuffle and balanced:
            counts = np.bincount(y.numpy())
            rare = int(np.argmin(counts))
            w = np.where(y.numpy() == rare, float(balanced), 1.0)
            smp = WeightedRandomSampler(weights=torch.tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
            return DataLoader(ds, batch_size=batch_size, sampler=smp)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, generator=g if shuffle else None)

    return prep('train', True), prep('valid', False), prep('test', False)


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing (复刻原实现)"""
    def __init__(self, gamma=3.0, label_smoothing=0.05, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        targets = nn.functional.one_hot(targets, num_classes).float()
        targets = (1 - self.label_smoothing) * targets + self.label_smoothing / num_classes
        logits = nn.functional.log_softmax(inputs, dim=-1)
        probs = torch.exp(logits)
        loss = -targets * ((1 - probs) ** self.gamma) * logits
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.sum(dim=-1).mean()


def compute_class_weights(labels, smooth_factor=0.01, exponent=0.7,
                          min_weight=0.5, max_weight=15.0):
    class_counts = np.bincount(labels)
    total = len(labels)
    smooth = int(smooth_factor * total)
    weights = []
    for count in class_counts:
        w = (total + smooth) / (count + smooth)
        w = w ** exponent
        weights.append(max(min_weight, min(w, max_weight)))
    weights = np.array(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_run(model, train_loader, valid_loader, is_cgf, lr=2e-4, epochs=40,
                  patience=20, device=DEVICE, verbose=False, resume_path=None,
                  select_metric='acc', class_weights=None):
    """早停/模型选择依据：验证集 select_metric（'acc' 准确率 | 'mf1' 宏F1）。
    resume_path: 若提供，每个 epoch 保存一次检查点，被杀死后可从断点继续。
    class_weights: 显式传入的类别权重（受控实验用，须与训练调用方冻结的权重一致；
    不传则维持旧行为——从训练加载器的一次遍历现抽样统计，断点恢复时会重新抽样）。"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if class_weights is None:
        train_labels = []
        for _, _, _, y in train_loader:
            train_labels.extend(y.numpy())
        class_weights = compute_class_weights(np.array(train_labels)).to(device)
    else:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(gamma=3.0, label_smoothing=0.05, class_weights=class_weights)

    start_epoch, best_score, best_state, patience_counter = 0, 0.0, None, 0

    def _restore(ck):
        nonlocal start_epoch, best_score, best_state, patience_counter
        start_epoch = ck['epoch'] + 1
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        scheduler.load_state_dict(ck['sched'])
        best_score = ck['best_acc']
        best_state = ck['best_state']
        patience_counter = ck['counter']
        if 'torch_rng' in ck:
            _r = ck['torch_rng']
            torch.set_rng_state(_r.cpu() if torch.is_tensor(_r) else _r)
        if 'numpy_rng' in ck:
            np.random.set_state(ck['numpy_rng'])
        if 'py_rng' in ck:
            import random as _py_random
            _py_random.setstate(ck['py_rng'])
        if 'class_weights' in ck:
            criterion.class_weights = ck['class_weights'].to(device)
        if verbose:
            print(f'  从检查点恢复：epoch {start_epoch}')

    if resume_path and os.path.exists(resume_path):
        _restore(torch.load(resume_path, map_location=device, weights_only=False))
    elif resume_path and os.path.exists(resume_path + '.tmp'):
        _restore(torch.load(resume_path + '.tmp', map_location=device, weights_only=False))

    for epoch in range(start_epoch, epochs):
        model.train()
        for batch in train_loader:
            text, audio, vision, y = batch[0], batch[1], batch[2], batch[-1]
            text, audio, vision, y = (x.to(device) for x in (text, audio, vision, y))
            tv = batch[3].to(device) if len(batch) == 5 else None
            optimizer.zero_grad()
            out = model(text, audio, vision, text_valid=tv)
            if is_cgf:
                main_loss = criterion(out['main_logits'], y)
                # -confidence 变体移除了置信度评估器及其辅助头，此时无辅助损失
                if 'text_logits' in out:
                    loss = main_loss + 0.5 * criterion(out['text_logits'], y)
                else:
                    loss = main_loss
            else:
                logits = out['main_logits'] if isinstance(out, dict) else out
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        all_pv, all_yv = [], []
        with torch.no_grad():
            for batch in valid_loader:
                text, audio, vision, y = batch[0], batch[1], batch[2], batch[-1]
                text, audio, vision, y = (x.to(device) for x in (text, audio, vision, y))
                tv = batch[3].to(device) if len(batch) == 5 else None
                out = model(text, audio, vision, text_valid=tv)
                logits = out['main_logits'] if isinstance(out, dict) else out
                _, pred = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                all_pv.extend(pred.cpu().numpy())
                all_yv.extend(y.cpu().numpy())
        valid_acc = 100 * correct / total
        if select_metric == 'mf1':
            from sklearn.metrics import f1_score as _f1
            score = _f1(all_yv, all_pv, average='macro')
        elif select_metric == 'last':
            score = float(epoch)  # 恒增：总是选最后一个 epoch，等效训满固定轮数
        else:
            score = valid_acc
        if verbose:
            print(f'  epoch {epoch+1}: valid_acc={valid_acc:.2f} score({select_metric})={score:.4f}')
        if score > best_score:
            best_score, best_state, patience_counter = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        if resume_path:
            os.makedirs(os.path.dirname(resume_path), exist_ok=True)
            tmp = resume_path + '.tmp'
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
                        'sched': scheduler.state_dict(), 'best_acc': best_score,
                        'best_state': best_state, 'counter': patience_counter,
                        'torch_rng': torch.get_rng_state(),
                        'numpy_rng': np.random.get_state(),
                        'py_rng': __import__('random').getstate(),
                        'class_weights': criterion.class_weights}, tmp)
            try:
                os.replace(tmp, resume_path)
            except OSError:
                pass  # 文件被临时占用时保留 .tmp，下个 epoch 或下次运行再恢复
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, test_loader, device=DEVICE, corruption=None, dump=False):
    """corruption: None 或 dict，如 {'missing': 'audio'} / {'noise': ('audio', sigma)}"""
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    all_y, all_pred, all_conf, all_gate = [], [], [], []
    for batch in test_loader:
        text, audio, vision, y = batch[0], batch[1], batch[2], batch[-1]
        tv = batch[3] if len(batch) == 5 else None
        text, audio, vision = text.clone(), audio.clone(), vision.clone()
        if corruption:
            if 'missing' in corruption:
                m = corruption['missing']
                if m in ('audio', 'both'):
                    audio = torch.zeros_like(audio)
                if m in ('vision', 'both'):
                    vision = torch.zeros_like(vision)
            if 'text_mask' in corruption:
                p = corruption['text_mask']
                if text.dim() == 3:  # 序列输入：按时间步掩码
                    mask = (torch.rand(text.shape[0], text.shape[1], 1) > p).float()
                else:                # 向量输入：按特征维掩码
                    mask = (torch.rand_like(text) > p).float()
                text = text * mask
            if 'noise' in corruption:
                mod, sigma = corruption['noise']
                tgt = text if mod == 'text' else (audio if mod == 'audio' else vision)
                std = tgt.std(dim=tuple(range(1, tgt.dim())), keepdim=True).clamp(min=1e-6)
                tgt = tgt + torch.randn_like(tgt) * sigma * std
                if mod == 'text':
                    text = tgt
                elif mod == 'audio':
                    audio = tgt
                else:
                    vision = tgt
        text, audio, vision = text.to(device), audio.to(device), vision.to(device)
        out = model(text, audio, vision,
                    text_valid=None if tv is None else tv.to(device))
        logits = out['main_logits'] if isinstance(out, dict) else out
        _, pred = torch.max(logits.data, 1)
        all_y.extend(y.numpy())
        all_pred.extend(pred.cpu().numpy())
        if dump and isinstance(out, dict):
            if 'confidence' in out:
                all_conf.extend(out['confidence'].cpu().numpy().tolist())
            if 'gate_weights' in out:
                all_gate.extend(out['gate_weights'].cpu().numpy().tolist())
    res = {
        'accuracy': float(accuracy_score(all_y, all_pred)),
        'macro_f1': float(f1_score(all_y, all_pred, average='macro')),
        'f1_per_class': [float(x) for x in f1_score(all_y, all_pred, average=None)],
        'labels': [int(x) for x in all_y],
        'preds': [int(x) for x in all_pred],
    }
    if all_conf:
        res['confidence'] = all_conf
    if all_gate:
        res['gate_weights'] = all_gate
    return res


class Progress:
    """可断点续跑的进度存取"""
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def has(self, key):
        return key in self.data

    def set(self, key, value):
        self.data[key] = value
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def get(self, key, default=None):
        return self.data.get(key, default)


def summarize(runs):
    """runs: list of dict with accuracy/macro_f1/f1_per_class -> mean±std"""
    acc = np.array([r['accuracy'] for r in runs])
    mf1 = np.array([r['macro_f1'] for r in runs])
    cls = np.array([r['f1_per_class'] for r in runs])
    return {
        'accuracy': float(acc.mean()), 'accuracy_std': float(acc.std()),
        'macro_f1': float(mf1.mean()), 'macro_f1_std': float(mf1.std()),
        'f1_per_class': cls.mean(axis=0).tolist(),
        'f1_per_class_std': cls.std(axis=0).tolist(),
        'n_runs': len(runs),
    }
