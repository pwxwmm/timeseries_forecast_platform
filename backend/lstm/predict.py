"""
LSTM Model Prediction Script
Independent prediction script for standalone execution and debugging

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
import numpy as np
import pandas as pd
import joblib

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from lstm.model import LSTMForecaster, GRUForecaster, create_model
from core.prometheus_api import PrometheusAPI, PrometheusConfig, create_feature_matrix

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """模型预测器"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_name: str):
        """
        加载训练好的模型

        Args:
            model_name: 模型名称（不包含扩展名）

        Returns:
            (model, feature_scaler, target_scaler, config)
        """
        logger.info(f"Loading model: {model_name}")

        # 加载配置
        config_path = self.model_dir / f"{model_name}_config.pkl"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = joblib.load(config_path)

        # 加载预处理器
        scaler_path = self.model_dir / f"{model_name}_scalers.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        scalers = joblib.load(scaler_path)
        feature_scaler = scalers["feature_scaler"]
        target_scaler = scalers["target_scaler"]

        # 创建模型
        model = create_model(
            model_type=config.get("model_type", "lstm"),
            input_dim=config.get("input_dim", 1),
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            output_dim=1,
            dropout=config.get("dropout", 0.2),
        ).to(self.device)

        # 加载模型权重
        model_path = self.model_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        logger.info(f"Model loaded successfully: {model_name}")
        return model, feature_scaler, target_scaler, config

    def load_latest_model(self):
        """加载最新的模型"""
        # 查找最新的模型文件
        model_files = list(self.model_dir.glob("model_*.pth"))
        if not model_files:
            raise FileNotFoundError("No model files found")

        # 按修改时间排序，取最新的
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model_name = latest_model.stem  # 去掉 .pth 扩展名

        return self.load_model(model_name)

    def get_latest_data(self, user: str, hours: int = 48) -> tuple:
        """
        获取最新的数据用于预测

        Args:
            user: 用户名
            hours: 获取最近多少小时的数据

        Returns:
            (features, target) 特征和目标数据
        """
        logger.info(f"Loading latest data for user: {user}, hours: {hours}")

        # 创建 Prometheus API 客户端
        prometheus_config = PrometheusConfig()
        api = PrometheusAPI(prometheus_config)

        # 获取历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

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
            return self._generate_synthetic_data(hours)

    def _generate_synthetic_data(self, hours: int) -> tuple:
        """生成合成数据用于测试"""
        np.random.seed(42)
        n_samples = hours

        # 创建带趋势和周期性的时间序列
        t = np.linspace(0, hours / 24, n_samples)
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

    def predict(
        self,
        model,
        feature_scaler,
        target_scaler,
        config,
        features: np.ndarray,
        steps: int = 1,
    ) -> np.ndarray:
        """
        进行预测

        Args:
            model: 训练好的模型
            feature_scaler: 特征预处理器
            target_scaler: 目标预处理器
            config: 模型配置
            features: 输入特征
            steps: 预测步数

        Returns:
            预测结果
        """
        logger.info(f"Making prediction for {steps} steps")

        # 数据预处理
        features_scaled = feature_scaler.transform(features)

        sequence_length = config.get("sequence_length", 24)

        # 检查数据长度
        if len(features_scaled) < sequence_length:
            raise ValueError(f"Input data must have at least {sequence_length} samples")

        predictions = []
        current_features = features_scaled.copy()

        for step in range(steps):
            # 取最后 sequence_length 个样本
            last_sequence = current_features[-sequence_length:]
            X = torch.tensor(
                last_sequence.reshape(1, sequence_length, -1), dtype=torch.float32
            ).to(self.device)

            # 预测
            with torch.no_grad():
                pred_scaled = model(X).cpu().numpy()

            # 反标准化
            pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            predictions.append(pred[0])

            # 更新特征（使用预测值作为下一个时间步的输入）
            # 这里简化处理，实际应用中可能需要更复杂的特征更新逻辑
            new_row = current_features[-1].copy()
            new_row[0] = pred_scaled[0]  # 更新主特征
            current_features = np.vstack([current_features, new_row])

        return np.array(predictions)

    def predict_and_save(
        self,
        user: str,
        model_name: str = None,
        steps: int = 1,
        save_to_prometheus: bool = True,
    ) -> dict:
        """
        预测并保存结果

        Args:
            user: 用户名
            model_name: 模型名称，如果为None则使用最新模型
            steps: 预测步数
            save_to_prometheus: 是否保存到Prometheus

        Returns:
            预测结果字典
        """
        try:
            # 加载模型
            if model_name:
                model, feature_scaler, target_scaler, config = self.load_model(
                    model_name
                )
            else:
                model, feature_scaler, target_scaler, config = self.load_latest_model()

            # 获取最新数据
            features, target = self.get_latest_data(user)

            # 进行预测
            predictions = self.predict(
                model, feature_scaler, target_scaler, config, features, steps
            )

            # 生成时间戳
            last_timestamp = datetime.now()
            timestamps = []
            for i in range(steps):
                next_time = last_timestamp + timedelta(hours=i + 1)
                timestamps.append(next_time.isoformat())

            result = {
                "user": user,
                "predictions": predictions.tolist(),
                "timestamps": timestamps,
                "model_name": model_name or "latest",
                "steps": steps,
            }

            logger.info(f"Prediction completed for user {user}:")
            for i, (pred, timestamp) in enumerate(zip(predictions, timestamps)):
                logger.info(f"  Step {i+1}: {pred:.4f} at {timestamp}")

            # 保存到Prometheus
            if save_to_prometheus:
                self._save_to_prometheus(user, predictions, timestamps)

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _save_to_prometheus(self, user: str, predictions: np.ndarray, timestamps: list):
        """保存预测结果到Prometheus Pushgateway"""
        try:
            from prometheus_api import PrometheusAPI, PrometheusConfig

            config = PrometheusConfig()
            api = PrometheusAPI(config)

            for pred, timestamp in zip(predictions, timestamps):
                success = api.push_prediction(
                    user=user, prediction_value=float(pred), prediction_horizon="1h"
                )

                if success:
                    logger.info(f"Prediction saved to Prometheus: {pred:.4f}")
                else:
                    logger.warning(
                        f"Failed to save prediction to Prometheus: {pred:.4f}"
                    )

        except Exception as e:
            logger.error(f"Failed to save to Prometheus: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Predict using trained LSTM model")
    parser.add_argument("--user", type=str, default="alice", help="User name")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (if not specified, use latest)",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of prediction steps"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Model directory"
    )
    parser.add_argument(
        "--no-prometheus", action="store_true", help="Do not save to Prometheus"
    )

    args = parser.parse_args()

    # 创建预测器
    predictor = ModelPredictor(model_dir=args.model_dir)

    try:
        # 进行预测
        result = predictor.predict_and_save(
            user=args.user,
            model_name=args.model_name,
            steps=args.steps,
            save_to_prometheus=not args.no_prometheus,
        )

        print("\nPrediction Results:")
        print(f"User: {result['user']}")
        print(f"Model: {result['model_name']}")
        print(f"Steps: {result['steps']}")
        print("\nPredictions:")
        for i, (pred, timestamp) in enumerate(
            zip(result["predictions"], result["timestamps"])
        ):
            print(f"  Step {i+1}: {pred:.4f} at {timestamp}")

        logger.info("Prediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
