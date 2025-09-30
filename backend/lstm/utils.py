"""
Utility Functions Module
Contains various helper functions and utility classes

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pickle

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = {}
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        if self.config_file.exists():
            try:
                import yaml

                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Config loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                self.config = {}
        else:
            logger.warning(f"Config file not found: {self.config_file}")
            self.config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save_config(self):
        """保存配置文件"""
        try:
            import yaml

            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Config saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")


class MetricsCalculator:
    """指标计算器"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算预测指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            指标字典
        """
        metrics = {}

        # 基本指标
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["r2"] = r2_score(y_true, y_pred)

        # 平均绝对百分比误差
        mask = y_true != 0
        if np.any(mask):
            metrics["mape"] = (
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            )
        else:
            metrics["mape"] = float("inf")

        # 对称平均绝对百分比误差
        if np.any(mask):
            metrics["smape"] = (
                np.mean(
                    2
                    * np.abs(y_true[mask] - y_pred[mask])
                    / (np.abs(y_true[mask]) + np.abs(y_pred[mask]))
                )
                * 100
            )
        else:
            metrics["smape"] = float("inf")

        # 方向准确率（趋势预测准确率）
        if len(y_true) > 1 and len(y_pred) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics["direction_accuracy"] = (
                np.mean(true_direction == pred_direction) * 100
            )
        else:
            metrics["direction_accuracy"] = 0.0

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """打印指标"""
        print("\nModel Performance Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            if metric in ["mse", "mae", "rmse"]:
                print(f"{metric.upper():20}: {value:.6f}")
            elif metric in ["r2"]:
                print(f"{metric.upper():20}: {value:.4f}")
            else:
                print(f"{metric.upper():20}: {value:.2f}%")


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_time_series(
        data: np.ndarray,
        min_length: int = 10,
        check_nan: bool = True,
        check_inf: bool = True,
    ) -> Dict[str, Any]:
        """
        验证时间序列数据

        Args:
            data: 时间序列数据
            min_length: 最小长度
            check_nan: 是否检查NaN值
            check_inf: 是否检查无穷值

        Returns:
            验证结果字典
        """
        result = {"is_valid": True, "issues": [], "stats": {}}

        # 检查长度
        if len(data) < min_length:
            result["is_valid"] = False
            result["issues"].append(f"Data too short: {len(data)} < {min_length}")

        # 检查NaN值
        if check_nan:
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                result["issues"].append(f"Found {nan_count} NaN values")
                result["is_valid"] = False

        # 检查无穷值
        if check_inf:
            inf_count = np.isinf(data).sum()
            if inf_count > 0:
                result["issues"].append(f"Found {inf_count} infinite values")
                result["is_valid"] = False

        # 统计信息
        if len(data) > 0:
            result["stats"] = {
                "length": len(data),
                "mean": np.nanmean(data),
                "std": np.nanstd(data),
                "min": np.nanmin(data),
                "max": np.nanmax(data),
                "nan_count": np.isnan(data).sum(),
                "inf_count": np.isinf(data).sum(),
            }

        return result

    @staticmethod
    def validate_features(features: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        验证特征数据

        Args:
            features: 特征数据
            target: 目标数据

        Returns:
            验证结果字典
        """
        result = {"is_valid": True, "issues": [], "stats": {}}

        # 检查维度匹配
        if len(features) != len(target):
            result["is_valid"] = False
            result["issues"].append(
                f"Feature and target length mismatch: {len(features)} vs {len(target)}"
            )

        # 检查特征维度
        if features.ndim != 2:
            result["is_valid"] = False
            result["issues"].append(f"Features should be 2D, got {features.ndim}D")

        # 检查目标维度
        if target.ndim != 1:
            result["is_valid"] = False
            result["issues"].append(f"Target should be 1D, got {target.ndim}D")

        # 统计信息
        if features.ndim == 2:
            result["stats"] = {
                "n_samples": features.shape[0],
                "n_features": features.shape[1],
                "feature_means": np.nanmean(features, axis=0).tolist(),
                "feature_stds": np.nanstd(features, axis=0).tolist(),
                "target_mean": np.nanmean(target),
                "target_std": np.nanstd(target),
            }

        return result


class ModelManager:
    """模型管理器"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict] = None):
        """
        保存模型

        Args:
            model: 模型对象
            model_name: 模型名称
            metadata: 元数据
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_name = f"{model_name}_{timestamp}"

        # 保存模型
        model_path = self.model_dir / f"{full_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # 保存元数据
        if metadata:
            metadata_path = self.model_dir / f"{full_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved: {full_name}")
        return full_name

    def load_model(self, model_name: str) -> Any:
        """
        加载模型

        Args:
            model_name: 模型名称

        Returns:
            模型对象
        """
        model_path = self.model_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded: {model_name}")
        return model

    def list_models(self) -> List[str]:
        """列出所有模型"""
        model_files = list(self.model_dir.glob("*.pkl"))
        return [f.stem for f in model_files]

    def delete_model(self, model_name: str):
        """删除模型"""
        model_path = self.model_dir / f"{model_name}.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"

        if model_path.exists():
            model_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        logger.info(f"Model deleted: {model_name}")


class Logger:
    """日志管理器"""

    @staticmethod
    def setup_logger(
        name: str, level: str = "INFO", log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        设置日志器

        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径

        Returns:
            日志器对象
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 清除现有处理器
        logger.handlers.clear()

        # 创建格式器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


def create_directory_structure(base_dir: str = "."):
    """创建项目目录结构"""
    base_path = Path(base_dir)

    directories = ["data", "models", "logs", "configs", "results", "notebooks"]

    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def generate_hash(data: Union[str, bytes]) -> str:
    """生成数据哈希值"""
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.md5(data).hexdigest()


def format_bytes(bytes_value: int) -> str:
    """格式化字节数"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """格式化时间长度"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        return f"{seconds/3600:.2f} hours"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    return a / b if b != 0 else default


# 示例使用
if __name__ == "__main__":
    # 测试配置管理器
    config_manager = ConfigManager()
    config_manager.set("model.hidden_dim", 128)
    config_manager.set("training.epochs", 100)
    print(f"Hidden dim: {config_manager.get('model.hidden_dim')}")
    print(f"Epochs: {config_manager.get('training.epochs')}")

    # 测试指标计算器
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])

    metrics = MetricsCalculator.calculate_metrics(y_true, y_pred)
    MetricsCalculator.print_metrics(metrics)

    # 测试数据验证器
    data = np.array([1, 2, np.nan, 4, 5])
    validation_result = DataValidator.validate_time_series(data)
    print(f"\nValidation result: {validation_result}")

    # 创建目录结构
    create_directory_structure()

    print("\nUtils module test completed!")
