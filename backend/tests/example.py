"""
Usage Example Script
Demonstrates how to use various modules for time series forecasting

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import sys
from pathlib import Path
import logging

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from lstm.data_loader import DataLoader
from lstm.model import create_model
from lstm.train import ModelTrainer
from lstm.predict import ModelPredictor
from lstm.utils import MetricsCalculator, Logger

# 配置日志
logger = Logger.setup_logger("example", level="INFO")


def example_synthetic_data():
    """使用合成数据的示例"""
    print("=" * 60)
    print("示例 1: 使用合成数据进行训练和预测")
    print("=" * 60)

    # 1. 创建数据加载器
    data_loader = DataLoader()

    # 2. 生成合成数据
    features, target = data_loader.generate_synthetic_data(
        n_samples=720, n_features=8, noise_level=0.1  # 30天 * 24小时
    )

    print(f"生成数据: features shape {features.shape}, target shape {target.shape}")

    # 3. 创建序列数据
    X, y = data_loader.create_sequences(
        features, target, sequence_length=24, prediction_steps=1
    )
    print(f"序列数据: X shape {X.shape}, y shape {y.shape}")

    # 4. 数据预处理
    X_scaled, y_scaled, feature_scaler, target_scaler = data_loader.preprocess_data(
        X.reshape(-1, X.shape[-1]), y, method="standard"
    )
    X_scaled = X_scaled.reshape(X.shape)

    # 5. 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        X_scaled, y_scaled, train_ratio=0.7, val_ratio=0.15
    )

    print(f"数据集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    # 6. 创建模型
    model = create_model(
        model_type="lstm",
        input_dim=features.shape[1],
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
    )

    print(f"模型信息: {model.get_model_info()}")

    # 7. 训练配置
    config = {
        "model_type": "lstm",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "num_layers": 2,
        "sequence_length": 24,
        "dropout": 0.2,
        "early_stopping_patience": 10,
    }

    # 8. 训练模型
    trainer = ModelTrainer(config)
    history = trainer.train_model(features, target)

    print(f"训练完成，最终损失: {history['val_loss'][-1]:.6f}")

    # 9. 进行预测
    predictor = ModelPredictor()

    # 使用最新数据进行预测
    latest_features, latest_target = data_loader.get_latest_data(
        "synthetic_user", hours=48
    )

    # 这里需要加载训练好的模型，简化处理
    print("预测功能需要训练好的模型，请使用 train.py 和 predict.py 脚本")


def example_model_comparison():
    """模型对比示例"""
    print("\n" + "=" * 60)
    print("示例 2: 不同模型类型对比")
    print("=" * 60)

    # 生成数据
    data_loader = DataLoader()
    features, target = data_loader.generate_synthetic_data(n_samples=200, n_features=5)

    # 创建不同模型
    models = {
        "LSTM": create_model("lstm", input_dim=5, hidden_dim=32, num_layers=1),
        "GRU": create_model("gru", input_dim=5, hidden_dim=32, num_layers=1),
    }

    print("模型参数对比:")
    for name, model in models.items():
        info = model.get_model_info()
        print(f"{name:10}: {info['total_parameters']:6d} parameters")

    # 测试前向传播
    import torch

    x = torch.randn(32, 24, 5)  # batch_size=32, seq_len=24, input_dim=5

    print("\n前向传播测试:")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"{name:10}: output shape {output.shape}")


def example_data_validation():
    """数据验证示例"""
    print("\n" + "=" * 60)
    print("示例 3: 数据验证")
    print("=" * 60)

    from utils import DataValidator
    import numpy as np

    # 测试数据
    test_data = np.array([1, 2, np.nan, 4, 5, np.inf, 7, 8])
    features = np.random.randn(8, 3)
    target = np.random.randn(8)

    # 时间序列验证
    ts_result = DataValidator.validate_time_series(test_data)
    print("时间序列验证结果:")
    print(f"  有效: {ts_result['is_valid']}")
    print(f"  问题: {ts_result['issues']}")
    print(f"  统计: {ts_result['stats']}")

    # 特征验证
    feature_result = DataValidator.validate_features(features, target)
    print("\n特征验证结果:")
    print(f"  有效: {feature_result['is_valid']}")
    print(f"  问题: {feature_result['issues']}")
    print(f"  统计: {feature_result['stats']}")


def example_metrics_calculation():
    """指标计算示例"""
    print("\n" + "=" * 60)
    print("示例 4: 指标计算")
    print("=" * 60)

    import numpy as np

    # 模拟预测结果
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9, 6.2, 6.8, 8.1, 8.9, 10.2])

    # 计算指标
    metrics = MetricsCalculator.calculate_metrics(y_true, y_pred)
    MetricsCalculator.print_metrics(metrics)


def main():
    """主函数"""
    print("时间序列预测平台 - 使用示例")
    print("=" * 60)

    try:
        # 运行示例
        example_synthetic_data()
        example_model_comparison()
        example_data_validation()
        example_metrics_calculation()

        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

        print("\n使用说明:")
        print("1. 训练模型: python train.py --user alice --epochs 50")
        print("2. 进行预测: python predict.py --user alice --steps 3")
        print("3. 查看帮助: python train.py --help")
        print("4. 查看帮助: python predict.py --help")

    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
