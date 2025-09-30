"""
Data Loading and Preprocessing Module
Specialized for data acquisition, cleaning and feature engineering

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

# 延迟导入以避免循环依赖
# from core.prometheus_api import PrometheusAPI, PrometheusConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器"""

    def __init__(self, config: Optional[Dict] = None, prometheus_api=None):
        self.config = config or {}
        self.prometheus_api = prometheus_api

    def load_prometheus_data(
        self, user: str, start_time: datetime, end_time: datetime, step: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """
        从 Prometheus 加载数据

        Args:
            user: 用户名
            start_time: 开始时间
            end_time: 结束时间
            step: 查询步长

        Returns:
            指标数据字典
        """
        logger.info(f"Loading Prometheus data for user: {user}")
        logger.info(f"Time range: {start_time} to {end_time}")

        try:
            if self.prometheus_api is None:
                # 延迟导入以避免循环依赖
                from core.prometheus_api import PrometheusAPI, PrometheusConfig

                self.prometheus_api = PrometheusAPI(PrometheusConfig())

            metrics = self.prometheus_api.get_storage_metrics(
                user=user, start=start_time, end=end_time, step=step
            )

            logger.info(f"Loaded {len(metrics)} metrics")
            for metric_name, df in metrics.items():
                if not df.empty:
                    logger.info(f"  {metric_name}: {len(df)} data points")
                else:
                    logger.warning(f"  {metric_name}: No data")

            return metrics

        except Exception as e:
            logger.error(f"Failed to load Prometheus data: {e}")
            raise

    def generate_synthetic_data(
        self, n_samples: int = 720, n_features: int = 5, noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成数据用于测试

        Args:
            n_samples: 样本数量
            n_features: 特征数量
            noise_level: 噪声水平

        Returns:
            (features, target) 特征和目标数据
        """
        logger.info(
            f"Generating synthetic data: {n_samples} samples, {n_features} features"
        )

        np.random.seed(42)

        # 时间序列
        t = np.linspace(0, n_samples / 24, n_samples)  # 以小时为单位

        # 目标序列：趋势 + 周期性 + 噪声
        trend = 0.05 * t
        daily_seasonal = 2 * np.sin(2 * np.pi * t / 24)
        weekly_seasonal = 1.5 * np.sin(2 * np.pi * t / (24 * 7))
        noise = noise_level * np.random.randn(n_samples)

        target = trend + daily_seasonal + weekly_seasonal + noise

        # 特征矩阵
        features = np.column_stack(
            [
                target,  # 主特征（目标序列）
                np.sin(2 * np.pi * t / 24),  # 小时特征
                np.cos(2 * np.pi * t / 24),  # 小时特征
                np.sin(2 * np.pi * t / (24 * 7)),  # 周特征
                np.cos(2 * np.pi * t / (24 * 7)),  # 周特征
            ]
        )

        # 如果需要的特征数更多，添加更多特征
        if n_features > 5:
            for i in range(5, n_features):
                features = np.column_stack([features, np.random.randn(n_samples) * 0.5])

        logger.info(
            f"Generated synthetic data: features shape {features.shape}, target shape {target.shape}"
        )
        return features, target

    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray,
        sequence_length: int,
        prediction_steps: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列训练数据

        Args:
            data: 输入特征数据 (n_samples, n_features)
            target: 目标数据 (n_samples,)
            sequence_length: 序列长度
            prediction_steps: 预测步数

        Returns:
            (X, y) 序列数据
        """
        logger.info(
            f"Creating sequences: seq_len={sequence_length}, pred_steps={prediction_steps}"
        )

        X, y = [], []
        for i in range(len(data) - sequence_length - prediction_steps + 1):
            # 输入序列
            x_seq = data[i : i + sequence_length]
            X.append(x_seq)

            # 目标值
            if prediction_steps == 1:
                y_seq = target[i + sequence_length]
            else:
                y_seq = target[
                    i + sequence_length : i + sequence_length + prediction_steps
                ]

            y.append(y_seq)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def preprocess_data(
        self, features: np.ndarray, target: np.ndarray, method: str = "standard"
    ) -> Tuple[np.ndarray, np.ndarray, object, object]:
        """
        数据预处理

        Args:
            features: 特征数据
            target: 目标数据
            method: 标准化方法 ("standard", "minmax", "robust")

        Returns:
            (features_scaled, target_scaled, feature_scaler, target_scaler)
        """
        logger.info(f"Preprocessing data using {method} scaling")

        # 选择标准化方法
        if method == "standard":
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
        elif method == "minmax":
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        elif method == "robust":
            feature_scaler = RobustScaler()
            target_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # 处理缺失值
        imputer = SimpleImputer(strategy="mean")
        features_imputed = imputer.fit_transform(features)

        # 标准化
        features_scaled = feature_scaler.fit_transform(features_imputed)
        target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        logger.info(f"Data preprocessing completed")
        return features_scaled, target_scaled, feature_scaler, target_scaler

    def create_features(
        self, metrics: Dict[str, pd.DataFrame], target_metric: str = "storage_used"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从多个指标创建特征矩阵

        Args:
            metrics: 指标数据字典
            target_metric: 目标指标名称

        Returns:
            (features, target) 特征矩阵和目标值
        """
        logger.info(f"Creating features from metrics, target: {target_metric}")

        if target_metric not in metrics:
            raise ValueError(f"Target metric '{target_metric}' not found in metrics")

        # 以目标指标的时间戳为基准
        base_df = metrics[target_metric].copy()
        base_df = base_df.set_index("timestamp")

        # 合并其他指标
        feature_columns = []
        for metric_name, df in metrics.items():
            if metric_name == target_metric:
                continue

            if not df.empty:
                df_indexed = df.set_index("timestamp")
                # 重采样到相同时间频率
                df_resampled = df_indexed.resample("1H").mean()
                base_df[f"{metric_name}_value"] = df_resampled["value"]
                feature_columns.append(f"{metric_name}_value")

        # 添加时间特征
        base_df["hour"] = base_df.index.hour
        base_df["day_of_week"] = base_df.index.dayofweek
        base_df["day_of_month"] = base_df.index.day
        base_df["month"] = base_df.index.month

        # 添加周期性特征
        base_df["hour_sin"] = np.sin(2 * np.pi * base_df["hour"] / 24)
        base_df["hour_cos"] = np.cos(2 * np.pi * base_df["hour"] / 24)
        base_df["day_sin"] = np.sin(2 * np.pi * base_df["day_of_week"] / 7)
        base_df["day_cos"] = np.cos(2 * np.pi * base_df["day_of_week"] / 7)

        feature_columns.extend(
            [
                "hour",
                "day_of_week",
                "day_of_month",
                "month",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
            ]
        )

        # 填充缺失值
        base_df = base_df.fillna(method="ffill").fillna(method="bfill")

        # 提取特征和目标
        features = base_df[feature_columns].values
        target = base_df["value"].values

        logger.info(
            f"Created features: shape {features.shape}, target shape {target.shape}"
        )
        return features, target

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分数据集

        Args:
            X: 特征数据
            y: 目标数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}")

        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        X_train = X[:train_size]
        X_val = X[train_size : train_size + val_size]
        X_test = X[train_size + val_size :]

        y_train = y[:train_size]
        y_val = y[train_size : train_size + val_size]
        y_test = y[train_size + val_size :]

        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_data(self, data: Dict, filepath: str):
        """
        保存数据到文件

        Args:
            data: 要保存的数据
            filepath: 文件路径
        """
        import joblib

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(data, filepath)
        logger.info(f"Data saved to: {filepath}")

    def load_data(self, filepath: str) -> Dict:
        """
        从文件加载数据

        Args:
            filepath: 文件路径

        Returns:
            加载的数据
        """
        import joblib

        data = joblib.load(filepath)
        logger.info(f"Data loaded from: {filepath}")
        return data


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建数据加载器
    loader = DataLoader()

    # 生成合成数据
    features, target = loader.generate_synthetic_data(n_samples=720, n_features=8)

    # 创建序列
    X, y = loader.create_sequences(
        features, target, sequence_length=24, prediction_steps=1
    )

    # 数据预处理
    X_scaled, y_scaled, feature_scaler, target_scaler = loader.preprocess_data(
        X.reshape(-1, X.shape[-1]), y, method="standard"
    )

    # 重新整形为序列格式
    X_scaled = X_scaled.reshape(X.shape)

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        X_scaled, y_scaled
    )

    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  y_test: {y_test.shape}")

    # 保存数据
    data_to_save = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }

    loader.save_data(data_to_save, "data/processed_data.pkl")
