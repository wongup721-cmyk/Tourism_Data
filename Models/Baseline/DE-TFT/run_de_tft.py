"""
DE-TFT 实验入口（示例）。

完整流程演示：
1. 加载 CSV 数据 + 归一化 + 滑动窗口
2. DE 优化 TFT 超参数
3. 用最优超参数训练最终模型
4. 在测试集上评估 RMSE 和 MAPE
5. 保存结果

用法：
    python run_de_tft.py --data data.csv --target target_column

    # 指定日期列（用于按时间顺序划分）：
    python run_de_tft.py --data data.csv --target target --date date

    # 调整超参数搜索范围：
    python run_de_tft.py --data data.csv --target target --de-pop-size 10 --de-max-iter 20
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# 支持从脚本所在目录直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import prepare_data
from de_optimizer import optimize_tft
from tft_model import TemporalFusionTransformer


# ============================================================
# 默认配置
# ============================================================
DEFAULT_CONFIG = {
    "de_pop_size": 6,
    "de_max_iter": 12,
    "max_epochs": 100,
    "early_stopping_patience": 15,
    "device": "auto",
}


def get_device(device_str: str = "auto") -> torch.device:
    """自动检测可用设备。"""
    if device_str != "auto":
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_final_model(
    model: TemporalFusionTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    batch_size: int,
    max_epochs: int = 100,
    patience: int = 15,
    device: torch.device = None,
    verbose: bool = True,
) -> TemporalFusionTransformer:
    """在全部训练数据上训练最终模型。"""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    n_train = len(X_train_t)
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
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

        avg_loss = total_loss / max(n_batches, 1)

        # 整个训练集评估用于早停
        model.eval()
        with torch.no_grad():
            full_pred = model(X_train_t)
            full_loss = criterion(full_pred, y_train_t).item()

        scheduler.step(full_loss)

        if full_loss < best_loss:
            best_loss = full_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(
    model: TemporalFusionTransformer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y,
    device: torch.device,
) -> dict:
    """
    在测试集上评估模型。

    返回 RMSE 和 MAPE（在反归一化后的原始空间计算）。
    """
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy()

    # 反归一化到原始空间
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    pred_original = scaler_y.inverse_transform(pred_scaled)

    rmse = np.sqrt(np.mean((y_test_original - pred_original) ** 2))

    # MAPE（跳过目标值为 0 的样本）
    y_flat = y_test_original.flatten()
    p_flat = pred_original.flatten()
    mask = np.abs(y_flat) > 0
    if mask.sum() == 0:
        mape = float('nan')
    else:
        mape = np.mean(np.abs((y_flat[mask] - p_flat[mask]) / y_flat[mask])) * 100

    mae = np.mean(np.abs(y_flat - p_flat))

    return {"rmse": float(rmse), "mape": float(mape), "mae": float(mae)}


def save_results(result: dict, results_dir: str = "results"):
    """保存实验结果。"""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = os.path.join(results_dir, f"de_tft_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    # CSV
    csv_path = os.path.join(results_dir, f"de_tft_results_{timestamp}.csv")
    row = {
        "RMSE": f"{result['test_rmse']:.4f}",
        "MAPE": f"{result['test_mape']:.4f}%",
        "MAE": f"{result['test_mae']:.4f}",
        "hidden_size": result["best_params"]["hidden_size"],
        "n_heads": result["best_params"]["n_heads"],
        "learning_rate": f"{result['best_params']['learning_rate']:.2e}",
        "dropout": result["best_params"]["dropout"],
        "n_lstm_layers": result["best_params"]["n_lstm_layers"],
        "batch_size": result["best_params"]["batch_size"],
        "DE_MAPE": f"{result['de_fitness']:.4f}%",
        "device": result["device"],
        "n_features": result["n_features"],
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n结果已保存: {json_path}")
    print(f"           {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DE-TFT: Differential Evolution optimized Temporal Fusion Transformer"
    )
    parser.add_argument("--data", required=True, help="CSV 数据文件路径")
    parser.add_argument("--target", default="target", help="目标列名（默认：target）")
    parser.add_argument("--date", default=None, help="日期列名（默认：None）")
    parser.add_argument("--de-pop-size", type=int, default=6, help="DE 种群大小（默认：6）")
    parser.add_argument("--de-max-iter", type=int, default=12, help="DE 最大迭代次数（默认：12）")
    parser.add_argument("--max-epochs", type=int, default=100, help="最大训练轮数（默认：100）")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值（默认：15）")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--results-dir", default="results", help="结果保存目录（默认：results）")
    args = parser.parse_args()

    print("=" * 60)
    print("DE-TFT 实验")
    print("=" * 60)
    print(f"数据: {args.data}")
    print(f"目标列: {args.target}")
    if args.date:
        print(f"日期列: {args.date}")
    print(f"配置: de_pop_size={args.de_pop_size}, de_max_iter={args.de_max_iter}, "
          f"max_epochs={args.max_epochs}, patience={args.patience}")

    device = get_device(args.device)
    print(f"设备: {device}")

    # 1. 准备数据
    data = prepare_data(args.data, target_col=args.target, date_col=args.date)

    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    scaler_y = data["scaler_y"]
    n_features = data["n_features"]

    # 2. 从训练集中分出验证集（最后 20%）用于 DE 优化
    n_train = len(X_train_full)
    n_val = int(n_train * 0.2)
    X_train_de = X_train_full[:-n_val] if n_val > 0 else X_train_full
    y_train_de = y_train_full[:-n_val] if n_val > 0 else y_train_full
    X_val_de = X_train_full[-n_val:] if n_val > 0 else X_train_full
    y_val_de = y_train_full[-n_val:] if n_val > 0 else y_train_full

    print(f"\n  DE 训练集: {len(X_train_de)}, DE 验证集: {len(X_val_de)}")

    # 3. DE 超参数优化
    best_params, best_fitness = optimize_tft(
        X_train_de, y_train_de,
        X_val_de, y_val_de,
        n_features=n_features,
        pop_size=args.de_pop_size,
        max_iter=args.de_max_iter,
        device=device,
        verbose=True,
    )

    # 4. 用最优参数训练最终模型
    print(f"\n  训练最终模型（全量训练数据）...")
    final_model = TemporalFusionTransformer(
        n_features=n_features,
        hidden_size=int(best_params["hidden_size"]),
        n_heads=int(best_params["n_heads"]),
        dropout=float(best_params["dropout"]),
        n_lstm_layers=int(best_params["n_lstm_layers"]),
    )

    final_model = train_final_model(
        final_model,
        X_train_full, y_train_full,
        learning_rate=float(best_params["learning_rate"]),
        batch_size=int(best_params["batch_size"]),
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        verbose=True,
    )

    # 5. 测试集评估
    metrics = evaluate_model(final_model, X_test, y_test, scaler_y, device)
    print(f"\n  测试集结果: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.4f}%, "
          f"MAE={metrics['mae']:.4f}")

    # 6. 汇总结果
    result = {
        "data_file": args.data,
        "target_col": args.target,
        "n_features": n_features,
        "n_train": len(X_train_full),
        "n_test": len(X_test),
        "best_params": best_params,
        "de_fitness": best_fitness,
        "test_rmse": metrics["rmse"],
        "test_mape": metrics["mape"],
        "test_mae": metrics["mae"],
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    save_results(result, args.results_dir)


if __name__ == "__main__":
    main()
