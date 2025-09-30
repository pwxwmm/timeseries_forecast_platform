# 部署指南

本文档介绍如何部署时间序列预测平台。

## 🚀 快速部署

### 使用启动脚本（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd timeseries_forecast_platform

# 运行启动脚本
./start.sh
```

启动脚本会自动：
- 检查环境依赖
- 创建虚拟环境
- 安装依赖
- 启动后端和前端服务

### 手动部署

#### 1. 后端部署

```bash
cd backend

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境
cp config.example.yaml config.yaml
# 编辑 config.yaml 文件

# 启动服务
python app.py
```

#### 2. 前端部署

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 或构建生产版本
npm run build
npm run preview
```

## 🐳 Docker 部署

### 使用 Docker Compose（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 单独构建镜像

```bash
# 构建后端镜像
cd backend
docker build -t timeseries-forecast-backend .

# 构建前端镜像
cd frontend
docker build -t timeseries-forecast-frontend .

# 运行容器
docker run -d -p 8000:8000 timeseries-forecast-backend
docker run -d -p 3000:80 timeseries-forecast-frontend
```

## ☁️ 云平台部署

### AWS 部署

#### 使用 ECS

1. 创建 ECR 仓库
2. 推送镜像到 ECR
3. 创建 ECS 任务定义
4. 创建 ECS 服务

```bash
# 构建并推送镜像
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com

docker build -t timeseries-forecast-backend ./backend
docker tag timeseries-forecast-backend:latest <account>.dkr.ecr.us-west-2.amazonaws.com/timeseries-forecast-backend:latest
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/timeseries-forecast-backend:latest
```

#### 使用 Elastic Beanstalk

1. 创建 `.ebextensions` 配置
2. 使用 EB CLI 部署

```bash
eb init
eb create production
eb deploy
```

### Google Cloud 部署

#### 使用 Cloud Run

```bash
# 构建镜像
gcloud builds submit --tag gcr.io/PROJECT-ID/timeseries-forecast-backend ./backend

# 部署到 Cloud Run
gcloud run deploy --image gcr.io/PROJECT-ID/timeseries-forecast-backend --platform managed --region us-central1
```

### Azure 部署

#### 使用 Container Instances

```bash
# 构建镜像
az acr build --registry <registry-name> --image timeseries-forecast-backend ./backend

# 创建容器实例
az container create --resource-group <resource-group> --name timeseries-backend --image <registry-name>.azurecr.io/timeseries-forecast-backend:latest --ports 8000
```

## 🔧 生产环境配置

### 环境变量

```bash
# 后端环境变量
export PROMETHEUS_URL=http://prometheus:9090
export PUSHGATEWAY_URL=http://pushgateway:9091
export LOG_LEVEL=INFO
export DEBUG=false

# 前端环境变量
export VITE_API_BASE_URL=https://api.yourdomain.com
```

### 数据库配置

生产环境建议使用 PostgreSQL 或 MySQL 替代 JSON 文件存储：

```python
# 在 backend/store.py 中添加数据库支持
import psycopg2
from sqlalchemy import create_engine

# 数据库连接
DATABASE_URL = "postgresql://user:password@localhost/timeseries_db"
engine = create_engine(DATABASE_URL)
```

### 缓存配置

使用 Redis 进行缓存：

```python
import redis

# Redis 连接
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

### 监控配置

#### Prometheus 监控

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'timeseries-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

#### Grafana 仪表板

创建 Grafana 仪表板监控：
- 系统性能指标
- 模型训练状态
- 预测准确率
- API 响应时间

### 日志配置

```python
# 在 backend/app.py 中配置日志
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
```

## 🔒 安全配置

### HTTPS 配置

使用 Let's Encrypt 获取 SSL 证书：

```bash
# 安装 certbot
sudo apt-get install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d yourdomain.com
```

### 防火墙配置

```bash
# UFW 防火墙配置
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### API 认证

添加 JWT 认证：

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

## 📊 性能优化

### 负载均衡

使用 Nginx 进行负载均衡：

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

### 数据库优化

```sql
-- 创建索引
CREATE INDEX idx_tasks_user ON tasks(user);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_models_user ON models(user);
```

### 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model_predictions(model_id: str):
    # 缓存模型预测结果
    pass
```

## 🔍 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep :8000
   lsof -i :8000
   ```

2. **内存不足**
   ```bash
   # 检查内存使用
   free -h
   top -p $(pgrep -f "python app.py")
   ```

3. **磁盘空间不足**
   ```bash
   # 检查磁盘空间
   df -h
   du -sh logs/ models/
   ```

### 日志分析

```bash
# 查看错误日志
grep ERROR logs/app.log

# 查看访问日志
tail -f logs/access.log

# 分析性能
grep "slow query" logs/app.log
```

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查数据库连接
curl http://localhost:8000/stats

# 检查 Prometheus 连接
curl http://localhost:8000/api/data/metrics/test
```

## 📈 扩展部署

### 水平扩展

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3
    environment:
      - WORKER_ID=${HOSTNAME}
```

### 垂直扩展

```yaml
# 增加资源限制
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## 🔄 更新部署

### 滚动更新

```bash
# 更新镜像
docker-compose pull

# 滚动更新
docker-compose up -d --no-deps backend
```

### 蓝绿部署

```bash
# 部署新版本到绿色环境
docker-compose -f docker-compose.green.yml up -d

# 切换流量
# 更新负载均衡器配置

# 停止蓝色环境
docker-compose -f docker-compose.blue.yml down
```

## 📞 支持

如有部署问题，请：

1. 查看日志文件
2. 检查配置文件
3. 验证网络连接
4. 提交 Issue
5. 联系技术支持

---

**注意**: 生产环境部署前请进行充分测试和安全评估。
