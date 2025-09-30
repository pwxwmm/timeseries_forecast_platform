#!/usr/bin/env python3
"""
Quick Start Script
Used for quick testing and demonstration of LSTM prediction functionality

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
from lstm.utils import MetricsCalculator, Logger

# 配置日志
logger = Logger.setup_logger("quick_start", level="INFO")


def quick_demo():
    """快速演示"""
    print("🚀 LSTM 时间序列预测 - 快速演示")
    print("=" * 50)

    # 1. 生成数据
    print("📊 生成合成数据...")
    data_loader = DataLoader()
    features, target = data_loader.generate_synthetic_data(
        n_samples=200, n_features=6, noise_level=0.1  # 约8天的数据
    )
    print(f"   数据形状: features {features.shape}, target {target.shape}")

    # 2. 创建序列
    print("🔄 创建时间序列...")
    X, y = data_loader.create_sequences(
        features, target, sequence_length=12, prediction_steps=1
    )
    print(f"   序列形状: X {X.shape}, y {y.shape}")

    # 3. 数据预处理
    print("⚙️  数据预处理...")
    X_scaled, y_scaled, feature_scaler, target_scaler = data_loader.preprocess_data(
        X.reshape(-1, X.shape[-1]), y, method="standard"
    )
    X_scaled = X_scaled.reshape(X.shape)
    print(f"   预处理完成")

    # 4. 创建模型
    print("🧠 创建 LSTM 模型...")
    model = create_model(
        model_type="lstm",
        input_dim=features.shape[1],
        hidden_dim=32,
        num_layers=1,
        output_dim=1,
        dropout=0.1,
    )
    print(f"   模型参数: {sum(p.numel() for p in model.parameters())} 个")

    # 5. 简单训练
    print("🏋️  开始训练...")
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 转换数据
    X_tensor = torch.tensor(X_scaled[:50], dtype=torch.float32)  # 只用前50个样本
    y_tensor = torch.tensor(y_scaled[:50], dtype=torch.float32)

    # 训练
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"   Epoch {epoch+1:2d}: Loss = {loss.item():.6f}")

    # 6. 测试预测
    print("🔮 测试预测...")
    model.eval()
    with torch.no_grad():
        # 使用最后几个样本进行预测
        test_X = X_tensor[-5:]
        test_y = y_tensor[-5:]

        predictions = model(test_X).squeeze()

        # 反标准化
        pred_original = target_scaler.inverse_transform(
            predictions.numpy().reshape(-1, 1)
        ).flatten()
        true_original = target_scaler.inverse_transform(
            test_y.numpy().reshape(-1, 1)
        ).flatten()

        # 计算指标
        metrics = MetricsCalculator.calculate_metrics(true_original, pred_original)

        print("   预测结果:")
        for i, (true, pred) in enumerate(zip(true_original, pred_original)):
            print(f"   样本 {i+1}: 真实值 = {true:.4f}, 预测值 = {pred:.4f}")

        print(f"\n   性能指标:")
        print(f"   MSE:  {metrics['mse']:.6f}")
        print(f"   MAE:  {metrics['mae']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")

    print("\n✅ 快速演示完成！")
    print("\n💡 提示:")
    print("   - 运行 'python train.py --help' 查看训练选项")
    print("   - 运行 'python predict.py --help' 查看预测选项")
    print("   - 运行 'python example.py' 查看更多示例")
    print("   - 运行 'python test_models.py' 运行完整测试")


def main():
    """主函数"""
    try:
        quick_demo()
    except Exception as e:
        logger.error(f"演示失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
