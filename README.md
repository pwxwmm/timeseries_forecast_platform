# 时间序列预测平台

基于 LSTM 的 Prometheus 数据预测平台，支持存储配额使用量预测和其他时间序列预测任务。

> 在云原生和微服务架构盛行的今天，如何提前预测系统资源使用情况，避免资源瓶颈和故障，成为了运维团队面临的重要挑战。特别是在智算场景下，某些特定监控场景很难做到准确性，而准确性的前提是告警噪音较少。

> 前段时间发生的一次生产级别故障让我深刻认识到传统预测方法的局限性。当时的业务场景是针对一批研究员做了组、用户权限隔离，同时针对商业存储也做了用户配额、组配额管理。我们使用的是Prometheus原生的`predict_linear`做单维度水位预测，但这个函数只能做单维度且线性的预测，在复杂场景下非常不准确。

> 比如用最近12小时的数据预测未来6小时是否会写满或超过阈值，但当用户进行瞬时快写和快删操作时，预测就完全失效了，最终导致存储写满，训练任务无法提交。这里需要说明的是，为什么没有设置固定阈值？因为高性能大规模商业存储成本极高，单个用户或组平时都维持在85%-95%左右，单位都是TB级别，所以1%的差异成本就很昂贵。

> 经过研究，我发现LSTM非常适合做多维度的时序预测。比如针对CV、CoGLLM等不同组和用户，可以根据**组、用户、存储类型、存储集群、使用时间（白天/夜间）**这5个维度进行聚合预测，然后将预测数据回写到Prometheus。

> 参考我之前调研的demo项目：
> - [PyTorch_pro Demo](https://github.com/pwxwmm/PyTorch_pro)
> - [技术博客](https://blog.csdn.net/qq_28513801/article/details/151657065)

> 基于以上实践，我整理了这个LSTM落地方案，提供给大家测试使用，解决Prometheus单一维度静态阈值、预测不准的问题。这是一个基于LSTM深度学习模型的时间序列预测平台，能够从Prometheus监控数据中学习模式，预测未来的资源使用趋势。前端使用Element UI，后端使用FastAPI。 这里简单说明下，不懂LLM、LSTM深度学习这些没关系，从SRE角度出发就当作是一个服务组件或者中间件或者工具，我们先会用，然后再学会调整。其实就是某个服务组件调优的过程，只是相对来说更加专业一些，涉及到更多的数学知识，不过在大模型的加持下，问题也不是很大。不过做这个的前提是告警体系和cmdb体系、自动化体系已经完善。


**Author**: mmwei3  
**Email**: mmwei3@iflytek.com, 1300042631@qq.com  
**Date**: 2025-08-27  
**Weather**: Cloudy
**博客地址**: https://mmwei.blog.csdn.net/article/details/152315032?fromshare=blogdetail&sharetype=blogdetail&sharerId=152315032&sharerefer=PC&sharesource=qq_28513801&sharefrom=from_link

## 🚀 功能特性

- **🧠 LSTM 深度学习模型**: 主要基于 LSTM 的时间序列预测，支持 GRU 和 Transformer（可选）
- **📊 Prometheus 集成**: 自动从 Prometheus 拉取历史数据，支持多种指标类型
- **🎯 多场景预测**: 支持存储配额、GPU资源、网络带宽等多种预测场景
- **🔄 实时预测**: 提供 REST API 接口进行实时预测
- **📈 可视化界面**: 现代化的 Vue.js 前端界面，支持图表展示
- **⚡ 高性能**: 基于 FastAPI 的高性能后端服务
- **🔧 灵活配置**: 支持多种模型参数和训练配置
- **🐳 容器化部署**: 支持 Docker 和 Docker Compose 部署
- **📝 完整文档**: 详细的 API 文档和使用说明

## 📁 项目结构

```
timeseries_forecast_platform/
├── backend/                    # 后端服务
│   ├── core/                  # 核心应用组件
│   │   ├── __init__.py        # 核心模块包
│   │   ├── app.py             # FastAPI 主应用
│   │   ├── store.py           # JSON 数据存储
│   │   ├── forecast.py        # 集成 LSTM 预测
│   │   └── prometheus_api.py  # Prometheus API 客户端
│   │
│   ├── lstm/                  # LSTM 模型组件
│   │   ├── __init__.py        # LSTM 模块包
│   │   ├── model.py           # LSTM/GRU/Transformer 模型
│   │   ├── train.py           # 模型训练脚本
│   │   ├── predict.py         # 模型预测脚本
│   │   ├── data_loader.py     # 数据加载和预处理
│   │   └── utils.py           # 工具函数和类
│   │
│   ├── tests/                 # 测试和示例脚本
│   │   ├── __init__.py        # 测试模块包
│   │   ├── test_models.py     # 模型测试脚本
│   │   ├── example.py         # 使用示例
│   │   └── quick_start.py     # 快速开始演示
│   │
│   ├── requirements.txt       # Python 依赖
│   ├── config.example.yaml    # 配置文件示例
│   ├── Dockerfile            # Docker 镜像构建
│   └── run_tests.py          # 测试运行脚本
│
└── frontend/                  # 前端应用
    ├── package.json          # 前端依赖
    ├── vite.config.js        # Vite 配置
    ├── index.html            # HTML 入口
    ├── Dockerfile            # 前端 Docker 构建
    └── src/
        ├── main.js           # Vue 应用入口
        ├── App.vue           # 主应用组件
        ├── api.js            # API 通信模块
        └── components/       # Vue 组件
            ├── Dashboard.vue     # 仪表板
            ├── TaskList.vue      # 任务列表
            ├── TaskEditor.vue    # 任务编辑器
            └── ModelManager.vue  # 模型管理
```

## 🛠️ 技术栈

### 后端技术栈
- **FastAPI**: 高性能异步 Web 框架
- **PyTorch**: 深度学习框架
  - **LSTM**: 主要模型，适合大多数时间序列预测任务
  - **GRU**: 轻量级替代方案，训练更快
  - **Transformer**: 可选模型，适合大规模数据
- **scikit-learn**: 机器学习工具库，用于数据预处理
- **pandas**: 数据处理和分析库
- **numpy**: 数值计算库
- **requests**: HTTP 客户端，用于 Prometheus API 调用
- **uvicorn**: ASGI 服务器
- **pydantic**: 数据验证和序列化

### 前端技术栈
- **Vue 3**: 渐进式 JavaScript 框架
- **Element Plus**: Vue 3 UI 组件库
- **Vite**: 快速构建工具
- **ECharts**: 数据可视化库
- **Axios**: HTTP 客户端
- **Vue Router**: 路由管理

### 部署和监控
- **Docker**: 容器化部署
- **Docker Compose**: 多容器编排
- **Prometheus**: 监控和告警系统
- **Pushgateway**: 指标推送网关
- **Nginx**: 反向代理和静态文件服务

## 🚀 快速开始

### 环境要求

#### 基础环境
- **Python**: 3.8+ (推荐 3.9+)
- **Node.js**: 16+ (推荐 18+)
- **Docker**: 20.10+ (可选，用于容器化部署)
- **Prometheus**: 2.30+ (用于数据源)

#### 硬件要求

**CPU 模式（推荐用于开发和测试）**:
- **CPU**: 4 核心以上
- **内存**: 8GB 以上
- **存储**: 10GB 可用空间
- **网络**: 稳定的网络连接

**GPU 模式（推荐用于生产环境）**:
- **GPU**: NVIDIA GPU，支持 CUDA 11.0+
  - 最低配置: Tesla T4 16GB / RTX 3080 10GB
  - 推荐配置: Tesla V100 32GB / RTX 4090 24GB
  - 高性能配置: Tesla A100 40GB/80GB / H100 80GB
  - 生产级配置: Tesla A100 80GB (多卡) / H100 80GB (多卡)
- **显存**: 16GB 以上（推荐 32GB+）
- **CPU**: 16 核心以上（推荐 32 核心+）
- **内存**: 32GB 以上（推荐 64GB+）
- **存储**: 100GB 可用空间（NVMe SSD 推荐）

**为什么需要 GPU？**
- LSTM 模型训练需要大量矩阵运算，GPU 可以显著加速训练过程
- 训练时间对比（基于 100 个 epoch）：
  - CPU (16核心): 2-4 小时
  - Tesla T4: 15-25 分钟
  - Tesla V100: 8-15 分钟
  - Tesla A100: 5-10 分钟
  - H100: 3-8 分钟
- 预测阶段 GPU 加速效果不明显，CPU 即可满足需求
- 生产级 GPU 支持更大的批次大小和更复杂的模型

```bash
Tue Sep 30 14:13:00 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A800-SXM4-80GB          On  | 00000000:3D:00.0 Off |                    0 |
| N/A   34C    P0              62W / 400W |      2MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A800-SXM4-80GB          On  | 00000000:42:00.0 Off |                    0 |
| N/A   31C    P0              62W / 400W |      2MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A800-SXM4-80GB          On  | 00000000:61:00.0 Off |                    0 |
| N/A   53C    P0             332W / 400W |  73895MiB / 81920MiB |     96%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A800-SXM4-80GB          On  | 00000000:67:00.0 Off |                    0 |
| N/A   41C    P0              65W / 400W |      2MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   4  NVIDIA A800-SXM4-80GB          On  | 00000000:AD:00.0 Off |                    0 |
| N/A   32C    P0              59W / 400W |      2MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA A800-SXM4-80GB          On  | 00000000:B1:00.0 Off |                    0 |
| N/A   36C    P0             102W / 400W |  67601MiB / 81920MiB |     10%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA A800-SXM4-80GB          On  | 00000000:D0:00.0 Off |                    0 |
| N/A   56C    P0             344W / 400W |  73897MiB / 81920MiB |     97%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA A800-SXM4-80GB          On  | 00000000:D3:00.0 Off |                    0 |
| N/A   65C    P0             356W / 400W |  73895MiB / 81920MiB |     95%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

```

### 方式一：本地开发部署

#### 1. 克隆项目

```bash
git clone <repository-url>
cd timeseries_forecast_platform
```

#### 2. 后端设置

```bash
cd backend

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 检查 GPU 支持（可选）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 安装依赖
pip install -r requirements.txt

# 如果需要 GPU 支持，安装 CUDA 版本的 PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确的安装命令
# 例如：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 复制配置文件
cp config.example.yaml config.yaml

# 编辑配置文件
vim config.yaml  # 或使用其他编辑器
```

#### 3. 前端设置

```bash
cd frontend

# 安装依赖
npm install
```

#### 4. 启动服务

**启动后端服务：**

```bash
cd backend
python core/app.py
```

或者使用 uvicorn：

```bash
uvicorn core.app:app --host 0.0.0.0 --port 8000 --reload
```

**启动前端服务：**

```bash
cd frontend
npm run dev
```

#### 5. 访问应用

- **前端界面**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

### 方式二：Docker 部署

#### 1. 使用 Docker Compose（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

#### 2. 单独构建镜像

```bash
# 构建后端镜像
cd backend
docker build -t timeseries-forecast-backend .

# 构建前端镜像
cd frontend
docker build -t timeseries-forecast-frontend .

# 运行容器
docker run -d -p 8000:8000 timeseries-forecast-backend
docker run -d -p 3000:3000 timeseries-forecast-frontend
```

## 📖 使用指南

### 1. 用户管理

首次使用需要创建用户账户：

```bash
# 通过 API 创建用户
curl -X POST "http://localhost:8000/users" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "alice",
       "email": "alice@example.com"
     }'
```

### 2. 创建预测任务

#### 通过 Web 界面

1. 打开前端界面 http://localhost:3000
2. 点击"创建任务"按钮
3. 填写任务信息：
   - **任务名称**: 描述性名称
   - **指标查询**: PromQL 格式的查询语句
   - **模型配置**: 训练参数设置
4. 提交任务，系统自动开始训练

#### 通过 API

```bash
curl -X POST "http://localhost:8000/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Storage Usage Prediction",
       "user": "alice",
       "metric_query": "storage_used_bytes{user=\"alice\"}",
       "config": {
         "sequence_length": 24,
         "prediction_steps": 1,
         "epochs": 100,
         "hidden_dim": 64,
         "num_layers": 2,
         "learning_rate": 0.001,
         "batch_size": 32,
         "dropout": 0.2
       }
     }'
```

### 3. 监控训练进度

在任务列表中查看训练状态：

- **pending**: 任务已创建，等待开始训练
- **running**: 模型正在训练中
- **completed**: 训练完成，可以进行预测
- **failed**: 训练失败，查看错误信息

### 4. 进行预测

#### 通过 Web 界面

1. 在任务列表中找到已完成的任务
2. 点击"预测"按钮
3. 设置预测参数（步数、时间范围等）
4. 查看预测结果和可视化图表

#### 通过 API

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "user": "alice",
       "metric_query": "storage_used_bytes{user=\"alice\"}",
       "prediction_steps": 3
     }'
```

### 5. 独立使用 LSTM 模块

#### 训练模型

```bash
cd backend/lstm

# 基本训练（使用默认 LSTM 模型）
python train.py --user alice --epochs 50

# 高级 LSTM 训练（生产级配置）
python train.py --user alice \
    --model-type lstm \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --hidden-dim 256 \
    --num-layers 4

# 使用 GRU 模型（推荐用于快速训练）
python train.py --user alice \
    --model-type gru \
    --hidden-dim 64 \
    --num-layers 2

# 使用 Transformer 模型（A100/H100 推荐配置）
python train.py --user alice \
    --model-type transformer \
    --d-model 512 \
    --nhead 16 \
    --num-layers 8 \
    --batch-size 256 \
    --epochs 200
```

#### 进行预测

```bash
# 基本预测
python predict.py --user alice --steps 3

# 指定模型预测
python predict.py --user alice \
    --model-name model_20231201_120000 \
    --steps 5

# 不保存到 Prometheus
python predict.py --user alice \
    --steps 1 \
    --no-prometheus
```

#### 运行测试和示例

```bash
cd backend

# 运行所有测试
python run_tests.py

# 运行快速演示
cd tests
python quick_start.py

# 运行使用示例
python example.py

# 运行模型测试
python test_models.py
```

## 🔧 配置说明

### 后端配置 (config.yaml)

```yaml
# Prometheus 配置
prometheus:
  base_url: "http://localhost:9090"
  pushgateway_url: "http://localhost:9091"
  timeout: 30

# 模型默认配置
model:
  sequence_length: 24
  prediction_steps: 1
  epochs: 100
  hidden_dim: 64
  num_layers: 2
  learning_rate: 0.001
  batch_size: 32
  dropout: 0.2
  early_stopping_patience: 10

# 数据配置
data:
  default_days: 30
  default_hours: 48
  step: "1h"

# 日志配置
logging:
  level: "INFO"
  file: "logs/app.log"
```

### 前端配置 (vite.config.js)

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

### Docker Compose 配置

```yaml
version: '3.8'

services:
  # 后端服务
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - PUSHGATEWAY_URL=http://pushgateway:9091
    volumes:
      - ./backend/data:/app/data
      - ./backend/models:/app/models
      - ./backend/logs:/app/logs
    depends_on:
      - prometheus
      - pushgateway

  # 前端服务
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  # Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  # Pushgateway
  pushgateway:
    image: prom/pushgateway:latest
    ports:
      - "9091:9091"

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend

volumes:
  prometheus_data:
```

## 🧠 模型选择说明

### 为什么主要使用 LSTM？

**LSTM 的优势**:
- **长期依赖**: 能够捕捉时间序列中的长期依赖关系
- **稳定性**: 训练过程相对稳定，不易出现梯度消失问题
- **成熟度**: 在时间序列预测领域应用广泛，技术成熟
- **资源需求**: 相比 Transformer，LSTM 对计算资源要求较低
- **可解释性**: 模型结构相对简单，便于理解和调试

**适用场景**:
- 存储使用量预测（日/周周期性）
- GPU 资源使用预测
- 网络流量预测
- 大多数单变量时间序列预测任务

### 为什么引入 GRU？

**GRU 的优势**:
- **计算效率**: 比 LSTM 参数更少，训练更快
- **性能相当**: 在很多任务上与 LSTM 性能相近
- **资源友好**: 适合资源受限的环境

**使用建议**:
- 当训练时间要求较高时选择 GRU
- 数据量较小时 GRU 可能表现更好
- 作为 LSTM 的轻量级替代方案

### 为什么引入 Transformer？（可选）

**Transformer 的优势**:
- **并行计算**: 训练过程可以并行化，理论上更快
- **注意力机制**: 能够关注重要的时间步
- **长序列**: 对超长序列的处理能力更强

**为什么不是主要选择？**
- **复杂度高**: 模型结构复杂，调试困难
- **资源需求大**: 需要更多内存和计算资源
- **过拟合风险**: 在小数据集上容易过拟合
- **时间序列特性**: 对于大多数时间序列任务，LSTM 已经足够

**使用建议**:
- 仅在以下情况考虑使用 Transformer：
  - 数据量非常大（> 100万样本）
  - 序列长度很长（> 1000 时间步）
  - 有充足的计算资源
  - LSTM 和 GRU 效果不理想

### 模型选择指南

| 场景 | 推荐模型 | GPU 配置 | 原因 |
|------|----------|----------|------|
| 存储配额预测 | LSTM | T4/V100 | 稳定可靠，适合周期性数据 |
| GPU 资源预测 | LSTM/GRU | V100/A100 | 需要捕捉使用模式 |
| 网络流量预测 | LSTM | T4/V100 | 处理突发流量变化 |
| 快速原型验证 | GRU | T4 | 训练速度快 |
| 大规模数据 | Transformer | A100/H100 | 并行计算优势 |
| 资源受限环境 | GRU | T4 | 参数少，效率高 |
| 企业级部署 | LSTM/Transformer | A100/H100 (多卡) | 高并发，高精度 |

### 生产级 GPU 配置建议

**Tesla T4 (16GB)**:
- 适合: 中小规模预测任务，开发测试
- 批次大小: 32-64
- 模型复杂度: 中等

**Tesla V100 (32GB)**:
- 适合: 大规模预测任务，生产环境
- 批次大小: 64-128
- 模型复杂度: 高

**Tesla A100 (40GB/80GB)**:
- 适合: 超大规模预测，多用户并发
- 批次大小: 128-256
- 模型复杂度: 很高

**H100 (80GB)**:
- 适合: 企业级部署，极致性能
- 批次大小: 256-512
- 模型复杂度: 最高

## 📊 支持的预测场景

### 1. 存储配额预测

**指标**: `storage_used_bytes{user="xxx"}`  
**用途**: 预测用户存储使用量，提前预警配额不足  
**应用场景**: 云存储、文件系统、数据库存储

```bash
# 训练存储预测模型
python train.py --user alice \
    --metric-query "storage_used_bytes{user='alice'}" \
    --epochs 100

# 预测未来24小时
python predict.py --user alice --steps 24
```

### 2. GPU 资源预测

**指标**: `gpu_memory_used{user="xxx"}`  
**用途**: 预测 GPU 显存使用量，优化资源调度  
**应用场景**: 机器学习训练、深度学习推理

```bash
# 训练 GPU 预测模型
python train.py --user alice \
    --metric-query "gpu_memory_used{user='alice'}" \
    --epochs 150

# 预测未来12小时
python predict.py --user alice --steps 12
```

### 3. 网络带宽预测

**指标**: `network_throughput{user="xxx"}`  
**用途**: 预测网络流量峰值，进行容量规划  
**应用场景**: CDN、网络监控、流量管理

```bash
# 训练网络预测模型
python train.py --user alice \
    --metric-query "network_throughput{user='alice'}" \
    --sequence-length 48

# 预测未来6小时
python predict.py --user alice --steps 6
```

### 4. CPU 使用率预测

**指标**: `cpu_usage_percent{user="xxx"}`  
**用途**: 预测 CPU 使用率，优化资源分配  
**应用场景**: 服务器监控、容器调度

### 5. 自定义指标

支持任何 Prometheus 指标的时间序列预测，包括：
- 业务指标（用户活跃度、订单量等）
- 系统指标（内存使用、磁盘 I/O 等）
- 应用指标（响应时间、错误率等）

## 🔌 API 接口详解

### 用户管理 API

#### 创建用户
```http
POST /users
Content-Type: application/json

{
  "username": "alice",
  "email": "alice@example.com"
}
```

#### 获取用户信息
```http
GET /users/{user_id}
```

#### 根据用户名获取用户
```http
GET /users/username/{username}
```

### 任务管理 API

#### 创建预测任务
```http
POST /tasks
Content-Type: application/json

{
  "name": "Storage Usage Prediction",
  "user": "alice",
  "metric_query": "storage_used_bytes{user=\"alice\"}",
  "config": {
    "sequence_length": 24,
    "prediction_steps": 1,
    "epochs": 100,
    "hidden_dim": 64,
    "num_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "dropout": 0.2
  }
}
```

#### 获取任务信息
```http
GET /tasks/{task_id}
```

#### 获取用户任务列表
```http
GET /tasks/user/{username}
```

#### 删除任务
```http
DELETE /tasks/{task_id}
```

### 模型管理 API

#### 获取模型信息
```http
GET /models/{model_id}
```

#### 获取用户模型列表
```http
GET /models/user/{username}
```

#### 获取任务模型列表
```http
GET /models/task/{task_id}
```

### 预测服务 API

#### 进行时间序列预测
```http
POST /predict
Content-Type: application/json

{
  "user": "alice",
  "metric_query": "storage_used_bytes{user=\"alice\"}",
  "prediction_steps": 3
}
```

**响应示例**:
```json
{
  "user": "alice",
  "metric_query": "storage_used_bytes{user=\"alice\"}",
  "predictions": [1024.5, 1050.2, 1075.8],
  "timestamps": [
    "2023-12-01T13:00:00",
    "2023-12-01T14:00:00",
    "2023-12-01T15:00:00"
  ],
  "confidence": 0.95
}
```

#### 获取用户指标数据
```http
GET /data/metrics/{user}?hours=24
```

### 系统信息 API

#### 健康检查
```http
GET /health
```

#### 平台统计信息
```http
GET /stats
```

#### 根路径
```http
GET /
```

## 🧪 测试指南

### 运行测试

```bash
cd backend

# 运行所有测试
python run_tests.py

# 运行特定测试
cd tests
python test_models.py    # 模型测试
python example.py        # 使用示例
python quick_start.py    # 快速演示
```

### 测试内容

1. **模型创建测试**: 验证 LSTM、GRU、Transformer 模型创建
2. **前向传播测试**: 验证模型输入输出形状
3. **数据加载测试**: 验证数据生成、预处理、序列创建
4. **工具函数测试**: 验证指标计算、数据验证等功能
5. **集成测试**: 验证完整的训练和预测流程

### 性能测试

```bash
# 使用 Apache Bench 进行 API 性能测试
ab -n 1000 -c 10 http://localhost:8000/health

# 使用 wrk 进行压力测试
wrk -t12 -c400 -d30s http://localhost:8000/health
```

## 📈 性能优化

### 模型优化

1. **数据预处理优化**
   - 使用归一化和标准化
   - 处理缺失值和异常值
   - 特征工程和特征选择

2. **模型结构优化**
   - 调整网络层数和隐藏单元数
   - 使用 Dropout 防止过拟合
   - 尝试不同的激活函数

3. **训练优化**
   - 使用学习率调度器
   - 实现早停机制
   - 使用梯度裁剪

4. **超参数调优**
   - 网格搜索或随机搜索
   - 使用贝叶斯优化
   - 交叉验证

### 系统优化

1. **缓存策略**
   - 缓存训练好的模型
   - 缓存预测结果
   - 使用 Redis 进行分布式缓存

2. **异步处理**
   - 使用后台任务进行模型训练
   - 异步数据加载
   - 非阻塞 I/O 操作

3. **资源管理**
   - GPU 内存优化
   - 批处理大小调整（T4: 32-64, V100: 64-128, A100: 128-256）
   - 多进程数据处理
   - 多 GPU 并行训练（A100/H100 多卡）

4. **监控告警**
   - 系统性能监控
   - 模型性能监控
   - 错误日志监控

## 🔍 故障排除

### 常见问题及解决方案

#### 1. Prometheus 连接失败

**问题**: 无法连接到 Prometheus 服务

**解决方案**:
```bash
# 检查 Prometheus 服务状态
curl http://localhost:9090/api/v1/query?query=up

# 检查配置文件
cat backend/config.yaml

# 测试网络连接
telnet localhost 9090
```

#### 2. 模型训练失败

**问题**: 训练过程中出现错误

**解决方案**:
```bash
# 检查数据是否充足
curl "http://localhost:8000/data/metrics/alice?hours=168"

# 检查日志
tail -f backend/logs/app.log

# 使用合成数据测试
cd backend/tests
python quick_start.py
```

#### 3. 预测结果异常

**问题**: 预测结果不准确或异常

**解决方案**:
```bash
# 检查模型是否训练完成
curl http://localhost:8000/models/user/alice

# 验证输入数据
curl "http://localhost:8000/data/metrics/alice?hours=48"

# 重新训练模型
cd backend/lstm
python train.py --user alice --epochs 200
```

#### 4. 内存不足

**问题**: 训练时出现内存错误

**解决方案**:
```bash
# 减少批次大小
python train.py --user alice --batch-size 16

# 减少序列长度
python train.py --user alice --sequence-length 12

# 使用更小的模型
python train.py --user alice --hidden-dim 32 --num-layers 1

# 使用 GRU 替代 LSTM（参数更少）
python train.py --user alice --model-type gru
```

#### 5. GPU 相关问题

**问题**: GPU 不可用或性能不佳

**解决方案**:
```bash
# 检查 CUDA 安装
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装 CUDA 版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果 GPU 内存不足，减少批次大小
# T4: batch-size 32-64
# V100: batch-size 64-128  
# A100: batch-size 128-256
python train.py --user alice --batch-size 64

# 强制使用 CPU
export CUDA_VISIBLE_DEVICES=""
python train.py --user alice
```

#### 6. 前端无法连接后端

**问题**: 前端页面无法加载数据

**解决方案**:
```bash
# 检查后端服务状态
curl http://localhost:8000/health

# 检查 CORS 配置
# 在 backend/core/app.py 中确认 CORS 设置

# 检查代理配置
# 在 frontend/vite.config.js 中确认代理设置
```

### 日志查看

```bash
# 查看应用日志
tail -f backend/logs/app.log

# 查看错误日志
grep ERROR backend/logs/app.log

# 查看训练日志
tail -f backend/logs/training.log

# 查看 Docker 日志
docker-compose logs -f backend
```

### 调试模式

```bash
# 启用调试模式
export DEBUG=true
export LOG_LEVEL=DEBUG

# 启动服务
cd backend
python core/app.py
```

## 🚀 部署指南

### 生产环境部署

#### 1. 环境准备

```bash
# 安装 Docker 和 Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 2. 配置生产环境

```bash
# 复制生产配置
cp docker-compose.yml docker-compose.prod.yml

# 编辑生产配置
vim docker-compose.prod.yml

# 设置环境变量
export PROMETHEUS_URL=http://your-prometheus:9090
export PUSHGATEWAY_URL=http://your-pushgateway:9091
```

#### 3. 启动服务

```bash
# 启动生产环境
docker-compose -f docker-compose.prod.yml up -d

# 检查服务状态
docker-compose -f docker-compose.prod.yml ps

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f
```

#### 4. 配置反向代理

```nginx
# nginx.conf
upstream backend {
    server backend:8000;
}

upstream frontend {
    server frontend:3000;
}

server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://frontend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 监控和告警

#### 1. Prometheus 监控

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'timeseries-forecast'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### 2. Grafana 仪表板

```json
{
  "dashboard": {
    "title": "Time Series Forecast Platform",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

## 🤝 贡献指南

### 开发环境设置

```bash
# Fork 项目到你的 GitHub 账户
git clone https://github.com/your-username/timeseries_forecast_platform.git
cd timeseries_forecast_platform

# 创建开发分支
git checkout -b feature/your-feature-name

# 设置开发环境
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖

cd ../frontend
npm install
```

### 代码规范

1. **Python 代码规范**
   - 使用 Black 进行代码格式化
   - 使用 flake8 进行代码检查
   - 使用 mypy 进行类型检查

```bash
# 格式化代码
black backend/

# 检查代码
flake8 backend/
mypy backend/
```

2. **JavaScript 代码规范**
   - 使用 ESLint 进行代码检查
   - 使用 Prettier 进行代码格式化

```bash
# 检查代码
npm run lint

# 格式化代码
npm run format
```

### 提交规范

```bash
# 提交信息格式
git commit -m "feat: add new prediction model"
git commit -m "fix: resolve memory leak in training"
git commit -m "docs: update API documentation"
```

提交类型：
- `feat`: 新功能
- `fix`: 修复问题
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建过程或辅助工具的变动

### Pull Request 流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的 Web 框架
- [Vue.js](https://vuejs.org/) - 渐进式 JavaScript 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Element Plus](https://element-plus.org/) - Vue 3 UI 组件库
- [Prometheus](https://prometheus.io/) - 监控和告警系统
- [ECharts](https://echarts.apache.org/) - 数据可视化库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- **GitHub Issues**: [提交问题](https://github.com/your-username/timeseries_forecast_platform/issues)
- **Email**: mmwei3@iflytek.com, 1300042631@qq.com
- **讨论区**: [GitHub Discussions](https://github.com/your-username/timeseries_forecast_platform/discussions)

## 🔮 未来规划

### 短期目标 (1-3 个月)

- [ ] 支持更多深度学习模型 (CNN-LSTM, Attention机制)
- [ ] 添加模型自动调优功能
- [ ] 实现分布式训练支持
- [ ] 添加实时数据流处理

### 中期目标 (3-6 个月)

- [ ] 支持多变量时间序列预测
- [ ] 添加异常检测功能
- [ ] 实现模型版本管理
- [ ] 添加 A/B 测试框架

### 长期目标 (6-12 个月)

- [ ] 支持联邦学习
- [ ] 添加自动特征工程
- [ ] 实现模型解释性分析
- [ ] 支持边缘计算部署

---

**注意**: 这是一个演示项目，生产环境使用前请进行充分测试和安全评估。建议在生产环境中添加认证、授权、数据加密等安全措施。