"""
LSTM Model Definition
Independent model file for easy reuse and maintenance

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMForecaster(nn.Module):
    """LSTM 时间序列预测模型"""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super(LSTMForecaster, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 计算 LSTM 输出维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 全连接层
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1
                n = param.size(0)
                param.data[(n // 4) : (n // 2)].fill_(1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)

        Returns:
            输出张量 (batch_size, output_dim)
        """
        # LSTM 前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 应用 dropout
        last_output = self.dropout_layer(last_output)

        # 全连接层
        output = self.fc(last_output)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


class GRUForecaster(nn.Module):
    """GRU 时间序列预测模型（可选替代方案）"""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super(GRUForecaster, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional

        # GRU 层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 计算 GRU 输出维度
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 全连接层
        self.fc = nn.Linear(gru_output_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)

        Returns:
            输出张量 (batch_size, output_dim)
        """
        # GRU 前向传播
        gru_out, hidden = self.gru(x)

        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]

        # 应用 dropout
        last_output = self.dropout_layer(last_output)

        # 全连接层
        output = self.fc(last_output)

        return output


class TransformerForecaster(nn.Module):
    """Transformer 时间序列预测模型（高级选项）"""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super(TransformerForecaster, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)

        Returns:
            输出张量 (batch_size, output_dim)
        """
        # 输入投影
        x = self.input_projection(x)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer 编码
        transformer_out = self.transformer(x)

        # 取最后一个时间步的输出
        last_output = transformer_out[:, -1, :]

        # 应用 dropout
        last_output = self.dropout_layer(last_output)

        # 输出投影
        output = self.output_projection(last_output)

        return output


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def create_model(model_type: str = "lstm", **kwargs):
    """
    创建模型的工厂函数

    Args:
        model_type: 模型类型 ("lstm", "gru", "transformer")
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    if model_type.lower() == "lstm":
        return LSTMForecaster(**kwargs)
    elif model_type.lower() == "gru":
        return GRUForecaster(**kwargs)
    elif model_type.lower() == "transformer":
        return TransformerForecaster(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# 示例使用
if __name__ == "__main__":
    # 创建 LSTM 模型
    lstm_model = LSTMForecaster(
        input_dim=3, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2
    )

    print("LSTM Model Info:")
    print(lstm_model.get_model_info())

    # 创建 GRU 模型
    gru_model = GRUForecaster(
        input_dim=3, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2
    )

    print("\nGRU Model Info:")
    print(f"Total parameters: {sum(p.numel() for p in gru_model.parameters())}")

    # 创建 Transformer 模型
    transformer_model = TransformerForecaster(
        input_dim=3, d_model=64, nhead=8, num_layers=2, output_dim=1, dropout=0.1
    )

    print("\nTransformer Model Info:")
    print(f"Total parameters: {sum(p.numel() for p in transformer_model.parameters())}")

    # 测试前向传播
    batch_size, seq_len, input_dim = 32, 24, 3
    x = torch.randn(batch_size, seq_len, input_dim)

    lstm_out = lstm_model(x)
    gru_out = gru_model(x)
    transformer_out = transformer_model(x)

    print(f"\nOutput shapes:")
    print(f"LSTM: {lstm_out.shape}")
    print(f"GRU: {gru_out.shape}")
    print(f"Transformer: {transformer_out.shape}")
