"""
LSTM Model Training Script
Independent training script for standalone execution and debugging

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from lstm.model import LSTMForecaster, GRUForecaster, create_model
from core.prometheus_api import PrometheusAPI, PrometheusConfig, create_feature_matrix
from core.store import JSONStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # 创建必要的目录
        self.model_dir = Path(config.get("model_dir", "models"))
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.data_dir = Path(config.get("data_dir", "data"))

        for dir_path in [self.model_dir, self.log_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)

    def load_data(self, user: str, days: int = 30) -> tuple:
        """
        加载训练数据

        Args:
            user: 用户名
            days: 历史天数

        Returns:
            (features, target) 特征和目标数据
        """
        logger.info(f"Loading data for user: {user}, days: {days}")

        # 创建 Prometheus API 客户端
        prometheus_config = PrometheusConfig()
        api = PrometheusAPI(prometheus_config)

        # 获取历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        try:
            metrics = api.get_storage_metrics(
                user=user, start=start_time, end=end_time, step="1h"
            )

            if "storage_used" not in metrics or metrics["storage_used"].empty:
                raise ValueError("No historical data available")

            # 创建特征矩阵
            features, target = create_feature_matrix(metrics, "storage_used")

            logger.info(
                f"Loaded data: features shape {features.shape}, target shape {target.shape}"
            )
            return features, target

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            # 生成模拟数据用于测试
            logger.info("Generating synthetic data for testing...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> tuple:
        """生成合成数据用于测试"""
        np.random.seed(42)
        n_samples = 720  # 30天 * 24小时

        # 创建带趋势和周期性的时间序列
        t = np.linspace(0, 30, n_samples)
        trend = 0.1 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 7) + np.sin(2 * np.pi * t / 24)
        noise = 0.5 * np.random.randn(n_samples)

        # 目标序列
        target = trend + seasonal + noise

        # 特征矩阵
        features = np.column_stack(
            [
                target,  # 主特征
                np.sin(2 * np.pi * t / 24),  # 小时特征
                np.cos(2 * np.pi * t / 24),  # 小时特征
                np.sin(2 * np.pi * t / 7),  # 周特征
                np.cos(2 * np.pi * t / 7),  # 周特征
            ]
        )

        return features, target

    def create_sequences(
        self, data: np.ndarray, target: np.ndarray, sequence_length: int
    ) -> tuple:
        """
        创建时间序列训练数据

        Args:
            data: 输入特征数据
            target: 目标数据
            sequence_length: 序列长度

        Returns:
            (X, y) 序列数据
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(target[i + sequence_length])

        return np.array(X), np.array(y)

    def train_model(self, features: np.ndarray, target: np.ndarray) -> dict:
        """
        训练模型

        Args:
            features: 特征数据
            target: 目标数据

        Returns:
            训练结果字典
        """
        logger.info("Starting model training...")

        # 数据预处理
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # 创建序列
        sequence_length = self.config.get("sequence_length", 24)
        X, y = self.create_sequences(features_scaled, target_scaled, sequence_length)

        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # 划分训练集和验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.get("batch_size", 32), shuffle=True
        )

        # 创建模型
        model = create_model(
            model_type=self.config.get("model_type", "lstm"),
            input_dim=features.shape[1],
            hidden_dim=self.config.get("hidden_dim", 64),
            num_layers=self.config.get("num_layers", 2),
            output_dim=1,
            dropout=self.config.get("dropout", 0.2),
        ).to(self.device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.config.get("learning_rate", 0.001)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # 训练历史
        history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        best_val_loss = float("inf")
        patience_counter = 0
        epochs = self.config.get("epochs", 100)
        early_stopping_patience = self.config.get("early_stopping_patience", 10)

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()

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
                self.save_model(model, feature_scaler, target_scaler, history)
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, "
                    f"Val Loss={val_loss:.6f}, LR={current_lr:.6f}"
                )

            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info("Model training completed!")
        return history

    def save_model(self, model, feature_scaler, target_scaler, history):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"

        # 保存模型
        model_path = self.model_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        # 保存预处理器
        scaler_path = self.model_dir / f"{model_name}_scalers.pkl"
        joblib.dump(
            {"feature_scaler": feature_scaler, "target_scaler": target_scaler},
            scaler_path,
        )

        # 保存训练历史
        history_path = self.model_dir / f"{model_name}_history.pkl"
        joblib.dump(history, history_path)

        # 保存配置
        config_path = self.model_dir / f"{model_name}_config.pkl"
        joblib.dump(self.config, config_path)

        logger.info(f"Model saved: {model_name}")
        return model_name

    def evaluate_model(self, model, features: np.ndarray, target: np.ndarray) -> dict:
        """评估模型性能"""
        logger.info("Evaluating model...")

        # 加载预处理器
        scaler_path = self.model_dir / "model_scalers.pkl"
        if scaler_path.exists():
            scalers = joblib.load(scaler_path)
            feature_scaler = scalers["feature_scaler"]
            target_scaler = scalers["target_scaler"]
        else:
            logger.warning("Scalers not found, using new ones")
            feature_scaler = StandardScaler()
            target_scaler = MinMaxScaler()
            features_scaled = feature_scaler.fit_transform(features)
            target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # 数据预处理
        features_scaled = feature_scaler.transform(features)
        target_scaled = target_scaler.transform(target.reshape(-1, 1)).flatten()

        # 创建序列
        sequence_length = self.config.get("sequence_length", 24)
        X, y_true = self.create_sequences(
            features_scaled, target_scaled, sequence_length
        )

        # 预测
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_pred_scaled = model(X_tensor).cpu().numpy()

        # 反标准化
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}

        logger.info(f"Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

        return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Train LSTM model for time series forecasting"
    )
    parser.add_argument("--user", type=str, default="alice", help="User name")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days of historical data"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "gru", "transformer"],
        help="Model type",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument(
        "--sequence-length", type=int, default=24, help="Sequence length"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    args = parser.parse_args()

    # 配置
    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "sequence_length": args.sequence_length,
        "dropout": args.dropout,
        "early_stopping_patience": args.early_stopping_patience,
        "model_dir": "models",
        "log_dir": "logs",
        "data_dir": "data",
    }

    # 创建训练器
    trainer = ModelTrainer(config)

    try:
        # 加载数据
        features, target = trainer.load_data(args.user, args.days)

        # 训练模型
        history = trainer.train_model(features, target)

        # 评估模型
        # 这里需要加载训练好的模型进行评估
        # metrics = trainer.evaluate_model(model, features, target)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
