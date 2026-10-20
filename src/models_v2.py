# -*- coding: utf-8 -*-
"""序列版 CGF 模型（忠实终版论文方法章：先交叉注意力、再平均池化聚合）

消融通过构造参数实现：
  use_cross_attn=False -> -cross-attention（拼接投影后直接池化融合）
  use_confidence=False -> -confidence evaluation（无置信度调制与增强，门控直接基于文本）
  use_gate=False       -> -gated fusion（拼接交互特征，不加门）
  辅助任务在训练循环中通过 is_cgf=False 关闭（-multi-task learning）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionSeq(nn.Module):
    """文本序列为 Query、音/视序列为 Key/Value 的交叉注意力 + FFN，
    输出沿时间维平均池化为单向量（对应论文式(3.8)-(3.11)）"""

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, query, key_value, key_padding_mask=None):
        # key_padding_mask: (B, T) True=填充位（审稿意见 padding-mask 敏感性实验）
        attn_out, _ = self.attn(query, key_value, key_value,
                                key_padding_mask=key_padding_mask)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        if key_padding_mask is None:
            return x.mean(dim=1)  # 平均池化聚合 (式3.11)
        valid = (~key_padding_mask).float().unsqueeze(-1)      # (B,T,1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


class CGFSeq(nn.Module):
    def __init__(self, text_dim=768, audio_dim=5, vision_dim=20, num_classes=3,
                 hidden_dim=256, num_heads=4, dropout=0.1,
                 use_cross_attn=True, use_confidence=True, use_gate=True,
                 confidence_mode='normal', constant_c=0.5,
                 confidence_stop_grad=False, use_padding_mask=False):
        # confidence_mode: 'normal' 用真实逐样本置信度；
        # 'constant' 受控消融：置信度固定为 constant_c（辅助分类头、辅助损失、
        # 增强网络与门控全部保留，仅置信度输入为常数；常数预先指定，不取自测试集）
        # confidence_stop_grad: True 时门控/增强路径使用 detach 后的置信度，
        # 文本分类头只受辅助任务监督（审稿意见 3 的机制对照）
        # use_padding_mask: True 时由原始输入推断有效时间步（非全零行），
        # 交叉注意力忽略填充 Key、时间平均池化只在有效步上求均值
        # （审稿意见 padding 敏感性实验；默认 False 与正文模型一致）
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_confidence = use_confidence
        self.use_gate = use_gate
        self.confidence_mode = confidence_mode
        self.constant_c = float(constant_c)
        self.use_padding_mask = bool(use_padding_mask)
        self.confidence_stop_grad = bool(confidence_stop_grad)
        self.num_classes = num_classes

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))

        if use_cross_attn:
            self.text_audio_attention = CrossModalAttentionSeq(hidden_dim, num_heads, dropout)
            self.text_visual_attention = CrossModalAttentionSeq(hidden_dim, num_heads, dropout)

        if use_confidence:
            self.confidence_evaluator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))
            self.confidence_modulator = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
            self.text_enhancer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout))

        if use_gate:
            self.audio_gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
            self.visual_gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, text, audio, vision, c_external=None, text_valid=None):
        # text/audio/vision: (B, T, D) 序列输入
        # c_external: 可选 (B,) 外部置信度（Shuffled-c 对照，纯推理）。
        # 提供时门控/增强分支改用该外部值（视为常数），out['confidence']
        # 仍报告本样本自身由文本分类分布算出的置信度，便于对照分析。
        # text_valid: 可选 (B,T) 有效时间步掩码（1=真实 token 位，来自
        # text_bert 的 attention_mask；padding-mask 敏感性实验专用）。
        # use_padding_mask=True 且未提供 text_valid 时退回"非全零行"判据。
        h_t = self.text_projection(text)        # (B,T,H)
        h_a = self.audio_projection(audio)
        h_v = self.visual_projection(vision)

        pad_mask = None
        if self.use_padding_mask:
            if text_valid is not None:
                valid = text_valid.bool()   # (B,T) 真实 token 位（attention_mask）
            else:
                # 有效步 = 三模态在该步并非全零（右零填充的逆）
                valid = ((text.abs().sum(-1) + audio.abs().sum(-1) +
                          vision.abs().sum(-1)) > 0)
            pad_mask = ~valid                    # (B,T) True=填充位

        if self.use_cross_attn:
            a_iv = self.text_audio_attention(h_t, h_a, key_padding_mask=pad_mask)
            v_iv = self.text_visual_attention(h_t, h_v, key_padding_mask=pad_mask)
        else:
            a_iv = h_a.mean(dim=1)
            v_iv = h_v.mean(dim=1)

        if pad_mask is None:
            t_vec = h_t.mean(dim=1)              # 文本单向量
        else:
            validf = (~pad_mask).float().unsqueeze(-1)
            t_vec = (h_t * validf).sum(dim=1) / validf.sum(dim=1).clamp(min=1.0)

        text_logits, confidence = None, None
        if self.use_confidence:
            text_logits = self.confidence_evaluator(t_vec)
            probs = F.softmax(text_logits, dim=-1)
            p_max, _ = probs.max(dim=-1)
            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
            max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32,
                                                 device=text.device))
            entropy_norm = entropy / max_entropy
            if self.confidence_mode == 'constant':
                confidence = torch.full_like(p_max, self.constant_c)
            else:
                confidence = 0.5 * (1 - entropy_norm) + 0.5 * p_max
            if c_external is not None:
                c_use = c_external.detach().to(confidence.device).float()
            else:
                c_use = confidence.detach() if self.confidence_stop_grad else confidence
            w_conf = self.confidence_modulator(c_use.unsqueeze(-1))
            enhanced = self.text_enhancer(t_vec)
            h_t_enh = enhanced * w_conf + t_vec * (1 - w_conf)
        else:
            h_t_enh = t_vec

        if self.use_gate:
            g_a = self.audio_gate_net(torch.cat([h_t_enh, a_iv], dim=-1))
            g_v = self.visual_gate_net(torch.cat([h_t_enh, v_iv], dim=-1))
            fused = torch.cat([t_vec, g_a * a_iv, g_v * v_iv], dim=-1)
            gate_weights = torch.cat([g_a, g_v], dim=-1)
        else:
            fused = torch.cat([t_vec, a_iv, v_iv], dim=-1)
            gate_weights = None

        fused_feat = self.fusion_layer(fused)
        main_logits = self.classifier(fused_feat)

        out = {'main_logits': main_logits}
        if text_logits is not None:
            out['text_logits'] = text_logits
        if confidence is not None:
            out['confidence'] = confidence
        if gate_weights is not None:
            out['gate_weights'] = gate_weights
        return out
