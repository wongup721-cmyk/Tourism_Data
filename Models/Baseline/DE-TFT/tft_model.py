"""
Temporal Fusion Transformer (TFT) — 纯 PyTorch 实现。

基于 Wu et al. (2023) DE-TFT 论文和 Lim et al. (2019) 原始 TFT 论文。
针对单序列、单步预测场景做了简化：
- 点预测 (MSE Loss)，不使用分位数回归
- 所有变量均为 past-observed（无 known-future 或 static 协变量）
- 保留核心组件：Variable Selection → LSTM → Multi-head Attention → GRN → Output
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 基础组件
# ============================================================

class GatedLinearUnit(nn.Module):
    """
    GLU(a) = σ(W₁·a + b₁) ⊙ (W₂·a + b₂)

    门控线性单元：sigmoid 门控控制信息流。
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN(a, c=None):
        η₁ = ELU(W₁·a + W₂·c + b)
        η₂ = GLU(η₁)
        输出 = LayerNorm(a + η₂)  如果 a 和 η₂ 维度一致，否则跳过残差

    门控残差网络：带 GLU 的非线性变换 + 残差连接。
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_fc = nn.Linear(context_size, hidden_size) if context_size else None
        self.elu = nn.ELU()

        self.glu = GatedLinearUnit(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # 如果输入输出维度不匹配，需要投影
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, a, c=None):
        hidden = self.fc1(a)
        if self.context_fc is not None and c is not None:
            hidden = hidden + self.context_fc(c)
        hidden = self.elu(hidden)
        hidden = self.glu(hidden)
        hidden = self.dropout(hidden)

        skip = a
        if self.skip is not None:
            skip = self.skip(a)

        return self.layer_norm(skip + hidden)


class GRN(GatedResidualNetwork):
    """GatedResidualNetwork 的简写别名。"""
    pass


class VariableSelectionNetwork(nn.Module):
    """
    变量选择网络（VSN）。

    对每个输入变量独立用 GRN 编码，再通过可学习的 Softmax 权重组合。
    这是 TFT 的核心创新：自动学习哪些变量重要。
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        """
        Args:
            input_size: 输入特征数
            hidden_size: GRN 隐藏维度
            dropout: Dropout 比例
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 每个输入变量一个独立的 GRN 编码器
        self.feature_grns = nn.ModuleList([
            GRN(1, hidden_size, hidden_size, dropout) for _ in range(input_size)
        ])

        # 权重网络：输入是所有变量的扁平拼接
        self.weight_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape

        # 对每个变量独立编码
        feature_outputs = []
        for i in range(self.input_size):
            feat = x[:, :, i:i+1]  # [batch, seq_len, 1]
            out = self.feature_grns[i](feat)  # [batch, seq_len, hidden_size]
            feature_outputs.append(out)

        stacked = torch.stack(feature_outputs, dim=2)  # [batch, seq_len, input_size, hidden_size]

        # 计算变量重要性权重（在时间维度上做平均池化）
        avg_feat = x.mean(dim=1)  # [batch, input_size]
        weights = self.weight_net(avg_feat)  # [batch, input_size]
        weights = weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, input_size, 1]

        # 加权组合
        combined = (stacked * weights).sum(dim=2)  # [batch, seq_len, hidden_size]

        return combined


class InterpretableMultiHeadAttention(nn.Module):
    """
    可解释多头注意力。

    与标准 MHA 相同，但额外返回注意力权重用于可解释性分析。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: [batch, seq_len, d_model]
            mask: 可选，[batch, seq_len, seq_len] 或 [seq_len, seq_len]

        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch, seq_len, _ = q.shape

        Q = self.W_q(q).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_k]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        output = self.W_o(attn_output)

        return output, attn_weights


# ============================================================
# TFT 主模型
# ============================================================

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer 简化实现。

    组件流程：
      Input [batch, seq, n_features]
        → Variable Selection Network  → [batch, seq, hidden]
        → LSTM Encoder               → [batch, seq, hidden]
        → Residual + LayerNorm
        → Multi-head Self-Attention  → [batch, seq, hidden]
        → Residual + LayerNorm
        → Last timestep              → [batch, hidden]
        → Output GRN                  → [batch, hidden]
        → Dense(1)                   → [batch, 1]
    """
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_lstm_layers: int = 2,
    ):
        """
        Args:
            n_features: 输入特征数
            hidden_size: 隐藏维度（d_model）
            n_heads: 注意力头数
            dropout: Dropout 比例
            n_lstm_layers: LSTM 层数
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(n_features, hidden_size, dropout)

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # 3. Post-LSTM LayerNorm
        self.post_lstm_ln = nn.LayerNorm(hidden_size)

        # 4. Multi-head Self-Attention
        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)

        # 5. Post-Attention LayerNorm
        self.post_attn_ln = nn.LayerNorm(hidden_size)

        # 6. Output GRN + 输出层
        self.output_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

        # 7. 全局 Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention: bool = False):
        """
        Args:
            x: [batch, seq_len, n_features]
            return_attention: 是否返回注意力权重（用于可解释性分析）

        Returns:
            pred: [batch, 1] 预测值
            attn_weights: (如果 return_attention=True) [batch, n_heads, seq_len, seq_len]
        """
        # Step 1: Variable Selection
        vsn_out = self.vsn(x)  # [batch, seq_len, hidden_size]

        # Step 2: LSTM Encoder
        lstm_out, _ = self.lstm(vsn_out)  # [batch, seq_len, hidden_size]
        lstm_out = self.dropout(lstm_out)

        # Step 3: Residual + LayerNorm
        temporal_features = self.post_lstm_ln(vsn_out + lstm_out)

        # Step 4: Self-Attention
        attn_out, attn_weights = self.attention(
            temporal_features, temporal_features, temporal_features
        )
        attn_out = self.dropout(attn_out)

        # Step 5: Residual + LayerNorm
        enriched = self.post_attn_ln(temporal_features + attn_out)

        # Step 6: 取最后一个时间步
        last_step = enriched[:, -1, :]  # [batch, hidden_size]

        # Step 7: Output GRN
        output_features = self.output_grn(last_step)

        # Step 8: 输出层
        pred = self.output_layer(output_features)

        if return_attention:
            return pred, attn_weights
        return pred

    def get_variable_importance(self, x):
        """
        返回变量重要性权重（用于可解释性分析）。

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            weights: [batch, n_features] 每个变量的重要性权重
        """
        avg_feat = x.mean(dim=1)  # [batch, n_features]
        weights = self.vsn.weight_net(avg_feat)  # [batch, n_features]
        return weights


# ============================================================
# 工具函数
# ============================================================

def create_tft_model(n_features: int, params: dict) -> TemporalFusionTransformer:
    """
    从参数字典创建 TFT 模型。

    Args:
        n_features: 输入特征数
        params: dict with keys hidden_size, n_heads, dropout, n_lstm_layers

    Returns:
        TFT 模型实例
    """
    return TemporalFusionTransformer(
        n_features=n_features,
        hidden_size=int(params["hidden_size"]),
        n_heads=int(params["n_heads"]),
        dropout=float(params["dropout"]),
        n_lstm_layers=int(params["n_lstm_layers"]),
    )
