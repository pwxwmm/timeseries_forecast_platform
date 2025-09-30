"""
Model Testing Script
Used for testing the functionality of various modules

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import sys
from pathlib import Path
import logging
import numpy as np
import torch

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from lstm.model import LSTMForecaster, GRUForecaster, create_model
from lstm.data_loader import DataLoader
from lstm.utils import MetricsCalculator, DataValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")

    # 测试 LSTM 模型
    lstm_model = LSTMForecaster(
        input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2
    )

    print(f"LSTM 模型信息: {lstm_model.get_model_info()}")

    # 测试 GRU 模型
    gru_model = GRUForecaster(
        input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2
    )

    print(f"GRU 模型参数数量: {sum(p.numel() for p in gru_model.parameters())}")

    # 测试工厂函数
    model = create_model("lstm", input_dim=5, hidden_dim=32, num_layers=1)
    print(f"工厂函数创建的模型类型: {type(model).__name__}")

    print("✓ 模型创建测试通过")


def test_model_forward():
    """测试模型前向传播"""
    print("\n测试模型前向传播...")

    # 创建测试数据
    batch_size, seq_len, input_dim = 32, 24, 5
    x = torch.randn(batch_size, seq_len, input_dim)

    # 测试 LSTM
    lstm_model = LSTMForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1)
    lstm_model.eval()

    with torch.no_grad():
        lstm_output = lstm_model(x)

    print(f"LSTM 输入形状: {x.shape}")
    print(f"LSTM 输出形状: {lstm_output.shape}")
    assert lstm_output.shape == (
        batch_size,
        1,
    ), f"LSTM 输出形状错误: {lstm_output.shape}"

    # 测试 GRU
    gru_model = GRUForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1)
    gru_model.eval()

    with torch.no_grad():
        gru_output = gru_model(x)

    print(f"GRU 输出形状: {gru_output.shape}")
    assert gru_output.shape == (batch_size, 1), f"GRU 输出形状错误: {gru_output.shape}"

    print("✓ 模型前向传播测试通过")


def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")

    data_loader = DataLoader()

    # 测试合成数据生成
    features, target = data_loader.generate_synthetic_data(n_samples=100, n_features=5)
    print(f"合成数据: features {features.shape}, target {target.shape}")

    # 测试序列创建
    X, y = data_loader.create_sequences(
        features, target, sequence_length=10, prediction_steps=1
    )
    print(f"序列数据: X {X.shape}, y {y.shape}")

    # 测试数据预处理
    X_scaled, y_scaled, feature_scaler, target_scaler = data_loader.preprocess_data(
        X.reshape(-1, X.shape[-1]), y, method="standard"
    )
    print(f"预处理后: X_scaled {X_scaled.shape}, y_scaled {y_scaled.shape}")

    # 测试数据划分
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        X_scaled.reshape(X.shape), y_scaled, train_ratio=0.7, val_ratio=0.15
    )
    print(f"数据划分: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    print("✓ 数据加载器测试通过")


def test_utils():
    """测试工具函数"""
    print("\n测试工具函数...")

    # 测试指标计算
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])

    metrics = MetricsCalculator.calculate_metrics(y_true, y_pred)
    print(f"计算的指标: {list(metrics.keys())}")

    # 测试数据验证
    test_data = np.array([1, 2, np.nan, 4, 5])
    validation_result = DataValidator.validate_time_series(test_data)
    print(f"数据验证结果: 有效={validation_result['is_valid']}")

    print("✓ 工具函数测试通过")


def test_integration():
    """集成测试"""
    print("\n集成测试...")

    # 1. 生成数据
    data_loader = DataLoader()
    features, target = data_loader.generate_synthetic_data(n_samples=200, n_features=5)

    # 2. 创建序列
    X, y = data_loader.create_sequences(
        features, target, sequence_length=20, prediction_steps=1
    )

    # 3. 数据预处理
    X_scaled, y_scaled, feature_scaler, target_scaler = data_loader.preprocess_data(
        X.reshape(-1, X.shape[-1]), y, method="standard"
    )
    X_scaled = X_scaled.reshape(X.shape)

    # 4. 创建模型
    model = create_model(
        "lstm", input_dim=features.shape[1], hidden_dim=32, num_layers=1
    )

    # 5. 简单训练测试
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 转换数据为张量
    X_tensor = torch.tensor(X_scaled[:10], dtype=torch.float32)  # 只用前10个样本测试
    y_tensor = torch.tensor(y_scaled[:10], dtype=torch.float32)

    # 训练几步
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # 6. 测试预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor[:5]).squeeze()
        actual = y_tensor[:5]

        # 计算指标
        metrics = MetricsCalculator.calculate_metrics(
            actual.numpy(), predictions.numpy()
        )
        print(f"测试指标: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")

    print("✓ 集成测试通过")


def main():
    """主测试函数"""
    print("开始运行模型测试...")
    print("=" * 50)

    try:
        test_model_creation()
        test_model_forward()
        test_data_loader()
        test_utils()
        test_integration()

        print("\n" + "=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
