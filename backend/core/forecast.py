"""
LSTM Time Series Forecasting Module
Supports storage quota usage prediction and other time series forecasting tasks

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

# 导入自定义模块
from lstm.model import LSTMForecaster, create_model
from lstm.utils import MetricsCalculator, DataValidator, ModelManager
from lstm.data_loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""

    input_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 24  # 使用过去24小时预测未来
    prediction_steps: int = 1  # 预测步数
    early_stopping_patience: int = 10


class LSTMForecaster(nn.Module):
    """LSTM 时间序列预测模型"""

    def __init__(self, config: ModelConfig):
        super(LSTMForecaster, self).__init__()
        self.config = config

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )

        # 全连接层
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # LSTM 前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 应用 dropout
        last_output = self.dropout(last_output)

        # 全连接层
        output = self.fc(last_output)

        return output


class TimeSeriesPredictor:
    """时间序列预测器"""

    def __init__(self, config: ModelConfig, model_dir: str = "models"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.is_trained = False

    def create_sequences(
        self, data: np.ndarray, target: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列训练数据

        Args:
            data: 输入特征数据 (n_samples, n_features)
            target: 目标值 (n_samples,)

        Returns:
            (X, y) 序列数据
        """
        if target is None:
            target = data[:, 0]  # 使用第一个特征作为目标

        X, y = [], []
        for i in range(
            len(data) - self.config.sequence_length - self.config.prediction_steps + 1
        ):
            # 输入序列
            x_seq = data[i : i + self.config.sequence_length]
            X.append(x_seq)

            # 目标值（单步或多步预测）
            if self.config.prediction_steps == 1:
                y_seq = target[i + self.config.sequence_length]
            else:
                y_seq = target[
                    i
                    + self.config.sequence_length : i
                    + self.config.sequence_length
                    + self.config.prediction_steps
                ]

            y.append(y_seq)

        return np.array(X), np.array(y)

    def prepare_data(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备训练数据

        Args:
            features: 特征数据
            target: 目标数据

        Returns:
            (X_tensor, y_tensor) 张量数据
        """
        # 数据标准化
        self.feature_scaler = StandardScaler()
        self.scaler = MinMaxScaler()

        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # 创建序列
        X, y = self.create_sequences(features_scaled, target_scaled)

        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if self.config.prediction_steps > 1:
            y_tensor = y_tensor.unsqueeze(-1)

        return X_tensor, y_tensor

    def train(
        self, features: np.ndarray, target: np.ndarray, validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            features: 特征数据
            target: 目标数据
            validation_split: 验证集比例

        Returns:
            训练历史
        """
        logger.info("Starting model training...")

        # 准备数据
        X, y = self.prepare_data(features, target)

        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # 初始化模型
        self.model = LSTMForecaster(self.config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # 训练历史
        history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            # 更新学习率
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # 记录历史
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.6f}"
                )

            # 早停
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.is_trained = True
        logger.info("Model training completed!")

        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            features: 输入特征

        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # 数据预处理
        features_scaled = self.feature_scaler.transform(features)

        # 创建序列
        if len(features_scaled) < self.config.sequence_length:
            raise ValueError(
                f"Input data must have at least {self.config.sequence_length} samples"
            )

        # 取最后 sequence_length 个样本
        last_sequence = features_scaled[-self.config.sequence_length :]
        X = torch.tensor(
            last_sequence.reshape(1, self.config.sequence_length, -1),
            dtype=torch.float32,
        )

        # 预测
        with torch.no_grad():
            pred_scaled = self.model(X).numpy()

        # 反标准化
        if self.config.prediction_steps == 1:
            pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        else:
            pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        return pred

    def evaluate(self, features: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            features: 特征数据
            target: 目标数据

        Returns:
            评估指标
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # 准备数据
        features_scaled = self.feature_scaler.transform(features)
        target_scaled = self.scaler.transform(target.reshape(-1, 1)).flatten()

        X, y_true = self.create_sequences(features_scaled, target_scaled)

        # 预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred_scaled = self.model(X_tensor).numpy()

        # 反标准化
        if self.config.prediction_steps == 1:
            y_pred = self.scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        else:
            y_pred = self.scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}

    def save_model(self, model_name: str = None):
        """保存模型"""
        if model_name is None:
            model_name = f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_path = self.model_dir / f"{model_name}.pth"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        feature_scaler_path = self.model_dir / f"{model_name}_feature_scaler.pkl"
        config_path = self.model_dir / f"{model_name}_config.json"

        # 保存模型
        torch.save(self.model.state_dict(), model_path)

        # 保存预处理器
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_scaler, feature_scaler_path)

        # 保存配置
        config_dict = {
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "output_dim": self.config.output_dim,
            "dropout": self.config.dropout,
            "sequence_length": self.config.sequence_length,
            "prediction_steps": self.config.prediction_steps,
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_name: str):
        """加载模型"""
        model_path = self.model_dir / f"{model_name}.pth"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        feature_scaler_path = self.model_dir / f"{model_name}_feature_scaler.pkl"
        config_path = self.model_dir / f"{model_name}_config.json"

        # 加载配置
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # 更新配置
        for key, value in config_dict.items():
            setattr(self.config, key, value)

        # 创建模型
        self.model = LSTMForecaster(self.config)
        self.model.load_state_dict(torch.load(model_path))

        # 加载预处理器
        self.scaler = joblib.load(scaler_path)
        self.feature_scaler = joblib.load(feature_scaler_path)

        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 3

    # 创建带趋势和周期性的时间序列
    t = np.linspace(0, 10, n_samples)
    trend = 0.1 * t
    seasonal = 2 * np.sin(2 * np.pi * t) + np.sin(4 * np.pi * t)
    noise = 0.5 * np.random.randn(n_samples)

    # 目标序列
    target = trend + seasonal + noise

    # 特征矩阵
    features = np.column_stack(
        [
            target,  # 主特征
            np.sin(2 * np.pi * t),  # 辅助特征1
            np.cos(2 * np.pi * t),  # 辅助特征2
        ]
    )

    # 创建预测器
    config = ModelConfig(input_dim=n_features, sequence_length=24, epochs=50)

    predictor = TimeSeriesPredictor(config)

    # 训练模型
    history = predictor.train(features, target)

    # 评估模型
    metrics = predictor.evaluate(features, target)
    print(f"Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # 预测
    last_features = features[-24:]  # 使用最后24个样本
    prediction = predictor.predict(last_features)
    print(f"Next prediction: {prediction[0]:.4f}")

    # 保存模型
    predictor.save_model("example_model")
