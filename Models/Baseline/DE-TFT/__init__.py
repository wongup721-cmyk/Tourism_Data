"""
DE-TFT: Differential Evolution optimized Temporal Fusion Transformer.

Wu et al. (2023) — Interpretable tourism volume forecasting with multivariate
time series under the impact of COVID-19.

本实现为纯 PyTorch 版本，适用于单序列、单步时间序列预测：
- 仅 past-observed 变量
- MSE 损失（非分位数回归）
- DE 仅优化 TFT 超参数
"""
