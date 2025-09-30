#!/bin/bash

# 时间序列预测平台启动脚本

echo "🚀 启动时间序列预测平台..."

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装，请先安装 Python 3.8+"
    exit 1
fi

# 检查 Node.js 是否安装
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js 16+"
    exit 1
fi

# 检查 npm 是否安装
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，请先安装 npm"
    exit 1
fi

echo "✅ 环境检查通过"

# 启动后端服务
echo "🔧 启动后端服务..."
cd backend

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建 Python 虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "📦 安装后端依赖..."
pip install -r requirements.txt

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "⚙️  创建配置文件..."
    cp config.example.yaml config.yaml
    echo "📝 请编辑 backend/config.yaml 文件配置 Prometheus 地址"
fi

# 启动后端服务（后台运行）
echo "🚀 启动后端服务 (端口 8000)..."
python app.py &
BACKEND_PID=$!

# 等待后端服务启动
sleep 5

# 启动前端服务
echo "🎨 启动前端服务..."
cd ../frontend

# 安装前端依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
fi

# 启动前端服务
echo "🚀 启动前端服务 (端口 3000)..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 服务启动完成！"
echo ""
echo "📱 前端界面: http://localhost:3000"
echo "🔧 后端 API: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo ""; echo "🛑 正在停止服务..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo "✅ 服务已停止"; exit 0' INT

# 保持脚本运行
wait
