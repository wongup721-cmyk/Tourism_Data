"""
差分进化 (Differential Evolution) 超参数优化模块。

使用 scipy.optimize.differential_evolution 优化 TFT 的 6 个关键超参数。
适应度函数 = 验证集 MAPE（越小越好）。
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from typing import Tuple
import time

from tft_model import TemporalFusionTransformer

# ============================================================
# DE 搜索空间定义
# ============================================================

# 6 个超参数的搜索范围
BOUNDS = [
    (16, 64),       # hidden_size（步长为 4 的整数）
    (1, 4),         # n_heads（整数）
    (-4, -1),       # log10(learning_rate): 1e-4 ~ 1e-1
    (0.05, 0.5),    # dropout
    (1, 3),         # n_lstm_layers（整数）
    (8, 64),        # batch_size（整数）
]

# 参数索引
IDX_HIDDEN = 0
IDX_HEADS = 1
IDX_LOG_LR = 2
IDX_DROPOUT = 3
IDX_LSTM = 4
IDX_BATCH = 5


def _params_to_config(params: np.ndarray, n_features: int) -> dict:
    """将 DE 的连续参数向量转换为模型配置。"""
    hidden_size = int(round(params[IDX_HIDDEN] / 4) * 4)  # 取 4 的倍数
    hidden_size = max(8, min(64, hidden_size))

    n_heads = int(round(params[IDX_HEADS]))
    n_heads = max(1, min(4, n_heads))

    # 确保 hidden_size 能被 n_heads 整除
    while hidden_size % n_heads != 0:
        n_heads -= 1
        if n_heads < 1:
            n_heads = 1
            hidden_size = (hidden_size // n_heads) * n_heads
            break

    learning_rate = 10 ** float(params[IDX_LOG_LR])
    dropout = float(params[IDX_DROPOUT])
    n_lstm_layers = int(round(params[IDX_LSTM]))
    n_lstm_layers = max(1, min(3, n_lstm_layers))
    batch_size = int(round(params[IDX_BATCH]))
    batch_size = max(8, min(64, batch_size))

    return {
        "hidden_size": hidden_size,
        "n_heads": n_heads,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "n_lstm_layers": n_lstm_layers,
        "batch_size": batch_size,
        "n_features": n_features,
    }


def train_and_evaluate(
    model: TemporalFusionTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    batch_size: int,
    max_epochs: int = 100,
    patience: int = 15,
    device: torch.device = None,
    verbose: bool = False,
) -> float:
    """
    训练 TFT 模型并返回验证集 MAPE。

    Args:
        model: TFT 模型实例
        X_train, y_train: 训练数据（归一化后）
        X_val, y_val: 验证数据（归一化后）
        learning_rate: 学习率
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        patience: Early stopping 耐心值
        device: 计算设备
        verbose: 是否打印训练详情

    Returns:
        val_mape: 验证集 MAPE（%），作为 DE 的适应度函数值。
                  注意：MAPE 在归一化空间计算，与真实空间的 MAPE 趋势一致，
                  适合作为优化目标。
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    n_train = len(X_train_t)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()

        # Mini-batch 训练
        indices = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

    # 加载最佳模型并计算 MAPE
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        y_val_np = y_val_t.cpu().numpy()
        pred_np = val_pred.cpu().numpy()

    # MAPE 计算（在归一化空间，因为 scaler 信息不在本函数内）
    # 归一化空间的 MAPE 与真实空间的 MAPE 趋势一致，适合做适应度函数
    mask = np.abs(y_val_np) > 1e-8
    if mask.sum() == 0:
        return 1e10

    mape = np.mean(np.abs((y_val_np[mask] - pred_np[mask]) / y_val_np[mask])) * 100

    return float(mape)


def optimize_tft(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    pop_size: int = 6,
    max_iter: int = 12,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[dict, float]:
    """
    使用差分进化算法优化 TFT 超参数。

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        n_features: 输入特征数
        pop_size: DE 种群大小（默认 6）
        max_iter: DE 最大迭代次数（默认 12）
        device: 计算设备
        verbose: 是否打印优化过程

    Returns:
        best_params: 最优参数字典
        best_fitness: 最优适应度值（MAPE）
    """
    if device is None:
        device = torch.device("cpu")

    eval_count = [0]  # 用列表以便在闭包中修改

    def objective(params):
        eval_count[0] += 1
        config = _params_to_config(params, n_features)

        if config["hidden_size"] % config["n_heads"] != 0:
            return 1e10

        model = TemporalFusionTransformer(
            n_features=n_features,
            hidden_size=config["hidden_size"],
            n_heads=config["n_heads"],
            dropout=config["dropout"],
            n_lstm_layers=config["n_lstm_layers"],
        )

        mape = train_and_evaluate(
            model,
            X_train, y_train,
            X_val, y_val,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            device=device,
            verbose=False,
        )

        if verbose and eval_count[0] % 5 == 0:
            print(f"  DE eval #{eval_count[0]}: MAPE={mape:.4f}%, "
                  f"h={config['hidden_size']}, heads={config['n_heads']}, "
                  f"lr={config['learning_rate']:.2e}, do={config['dropout']:.2f}, "
                  f"lstm={config['n_lstm_layers']}, bs={config['batch_size']}")

        return mape

    if verbose:
        print(f"\n  开始 DE 优化: pop_size={pop_size}, max_iter={max_iter}, "
              f"特征数={n_features}")
        start_time = time.time()

    result = differential_evolution(
        objective,
        BOUNDS,
        popsize=pop_size,
        maxiter=max_iter,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=None,  # 不固定种子，每次运行产生不同结果
        disp=False,
        polish=False,  # 不做局部精化，节省时间
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"  DE 优化完成，耗时 {elapsed:.0f}s，共 {eval_count[0]} 次评估")
        print(f"  最优 MAPE: {result.fun:.4f}%")

    best_params = _params_to_config(result.x, n_features)

    if verbose:
        print(f"  最优参数: {best_params}")

    return best_params, float(result.fun)
