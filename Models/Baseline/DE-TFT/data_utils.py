"""
数据加载与预处理模块。

正确的归一化协议：
1. 先按时间顺序划分 train/test（前 70% 训练，后 30% 测试）
2. Scaler 仅在训练集上 fit
3. 用训练集的 scaler transform 测试集
4. 构建滑动窗口（window=30, horizon=1）
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

# ============================================================
# 配置常量
# ============================================================
WINDOW = 30        # 输入序列长度（周/天/任意时间单位）
HORIZON = 1        # 预测步长
TRAIN_RATIO = 0.7  # 训练集比例


def load_data(
    filepath: str,
    target_col: str = "target",
    date_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CSV 文件加载时间序列数据。

    Args:
        filepath: CSV 文件路径
        target_col: 目标列名（预测目标）
        date_col: 日期列名（可选，不传则自动生成序号）

    Returns:
        X: 特征数组 [n_samples, n_features]（不含目标列）
        y: 目标数组 [n_samples, 1]
        dates: 日期数组 [n_samples]
    """
    df = pd.read_csv(filepath)

    # 确定特征列：除目标列和日期列之外的所有列
    exclude_cols = {target_col}
    if date_col and date_col in df.columns:
        exclude_cols.add(date_col)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[[target_col]].values.astype(np.float32)

    if date_col and date_col in df.columns:
        dates = df[date_col].values
    else:
        dates = np.arange(len(y_raw)).astype(str)

    # 去重检测
    X_raw, y_raw, dates = _deduplicate(X_raw, y_raw, dates)

    return X_raw, y_raw, dates


def _deduplicate(X: np.ndarray, y: np.ndarray, dates: np.ndarray):
    """去除重复样本（基于目标值 + 日期完全重复）。"""
    n_before = len(y)
    seen = set()
    keep_idx = []
    for i in range(len(dates)):
        key = (str(dates[i]), float(y[i]))
        if key not in seen:
            seen.add(key)
            keep_idx.append(i)
    if len(keep_idx) < n_before:
        print(f"  去重: {n_before} -> {len(keep_idx)} 样本（移除 {n_before - len(keep_idx)} 条重复）")
    return X[keep_idx], y[keep_idx], dates[keep_idx]


def train_test_split_time(
    X: np.ndarray, y: np.ndarray, dates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按时间顺序划分训练集和测试集（不打乱）。

    默认前 TRAIN_RATIO 作为训练集，后面作为测试集。

    Returns:
        X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    n = len(y)
    train_size = int(n * TRAIN_RATIO)

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    dates_train = dates[:train_size]
    dates_test = dates[train_size:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def normalize_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    正确的归一化流程（避免数据泄露）：
    - 仅在训练集上 fit scaler
    - 用训练集的 scaler transform 测试集

    Returns:
        X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train)

    X_test_sc = scaler_X.transform(X_test)
    y_test_sc = scaler_y.transform(y_test)

    return X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y


def create_sliding_windows(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从时间序列数据构建滑动窗口。

    对每条样本：
      - 输入: X[i : i+WINDOW]  形状 [WINDOW, n_features]
      - 输出: y[i+WINDOW]       形状 [1]

    Args:
        X: 特征 [n_samples, n_features]
        y: 目标 [n_samples, 1]

    Returns:
        X_windows: [n_windows, WINDOW, n_features]
        y_windows: [n_windows, 1]
    """
    X_list, y_list = [], []
    for i in range(len(X) - WINDOW):
        X_list.append(X[i : i + WINDOW])
        y_list.append(y[i + WINDOW])

    X_windows = np.array(X_list, dtype=np.float32)
    y_windows = np.array(y_list, dtype=np.float32)

    return X_windows, y_windows


def prepare_data(
    filepath: str,
    target_col: str = "target",
    date_col: Optional[str] = None,
) -> dict:
    """
    一站式数据准备：加载 → 划分 → 归一化 → 滑动窗口。

    Args:
        filepath: CSV 文件路径
        target_col: 目标列名
        date_col: 日期列名（可选）

    Returns:
        dict with keys:
            X_train, y_train, X_test, y_test: 滑动窗口后的数据
            scaler_y: 目标变量的 scaler（用于反归一化）
            dates_train, dates_test: 日期标签
            train_ratio: 训练集比例
            n_features: 输入特征数
    """
    print(f"\n{'='*60}")
    print(f"加载数据: {filepath}")
    print(f"{'='*60}")

    # 1. 加载数据
    X, y, dates = load_data(filepath, target_col, date_col)
    n_features = X.shape[1]
    print(f"  总样本数: {len(y)}, 特征数: {n_features}")
    print(f"  日期范围: {dates[0]} ~ {dates[-1]}")

    # 2. 按时间划分
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, dates_train, dates_test = \
        train_test_split_time(X, y, dates)
    train_size = len(y_train_raw)
    test_size = len(y_test_raw)
    print(f"  训练集: {train_size} ({train_size/len(y)*100:.0f}%), "
          f"测试集: {test_size} ({test_size/len(y)*100:.0f}%)")
    print(f"  训练集日期: {dates_train[0]} ~ {dates_train[-1]}")
    print(f"  测试集日期: {dates_test[0]} ~ {dates_test[-1]}")

    # 3. 归一化（仅在训练集上 fit）
    X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y = \
        normalize_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

    # 4. 构建滑动窗口
    X_train_win, y_train_win = create_sliding_windows(X_train_sc, y_train_sc)
    X_test_win, y_test_win = create_sliding_windows(X_test_sc, y_test_sc)

    # 滑动窗口后调整日期
    dates_train_win = dates_train[WINDOW:]
    dates_test_win = dates_test[WINDOW:]

    print(f"  滑动窗口后 - 训练集: {len(X_train_win)}, 测试集: {len(X_test_win)}")

    return {
        "X_train": X_train_win,
        "y_train": y_train_win,
        "X_test": X_test_win,
        "y_test": y_test_win,
        "scaler_y": scaler_y,
        "dates_train": dates_train_win,
        "dates_test": dates_test_win,
        "train_ratio": TRAIN_RATIO,
        "n_features": n_features,
    }
