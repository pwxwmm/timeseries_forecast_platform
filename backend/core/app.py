"""
FastAPI Main Application
Provides REST API interfaces for time series forecasting

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from .prometheus_api import PrometheusAPI, PrometheusConfig, create_feature_matrix
from .forecast import TimeSeriesPredictor, ModelConfig
from .store import JSONStore, Task, Model, User

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
store = None
prometheus_api = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global store, prometheus_api

    # 启动时初始化
    logger.info("Starting application...")
    store = JSONStore()
    prometheus_config = PrometheusConfig()
    prometheus_api = PrometheusAPI(prometheus_config)
    logger.info("Application started successfully")

    yield

    # 关闭时清理
    logger.info("Shutting down application...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Time Series Forecast Platform",
    description="基于 LSTM 的 Prometheus 数据预测平台",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic 模型
class TaskCreate(BaseModel):
    name: str
    user: str
    metric_query: str
    config: Dict[str, Any]


class TaskResponse(BaseModel):
    id: str
    name: str
    user: str
    metric_query: str
    status: str
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    model_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ModelCreate(BaseModel):
    name: str
    user: str
    task_id: str
    model_type: str
    config: Dict[str, Any]


class ModelResponse(BaseModel):
    id: str
    name: str
    user: str
    task_id: str
    model_type: str
    status: str
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None
    file_path: Optional[str] = None


class PredictionRequest(BaseModel):
    user: str
    metric_query: str
    model_id: Optional[str] = None
    prediction_steps: int = 1


class PredictionResponse(BaseModel):
    user: str
    metric_query: str
    predictions: List[float]
    timestamps: List[str]
    confidence: Optional[float] = None


class UserCreate(BaseModel):
    username: str
    email: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: str
    last_login: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


# 依赖注入
def get_store() -> JSONStore:
    return store


def get_prometheus_api() -> PrometheusAPI:
    return prometheus_api


# 后台任务
async def train_model_task(task_id: str, model_id: str):
    """后台训练模型任务"""
    try:
        logger.info(f"Starting model training for task {task_id}")

        # 更新任务状态
        store.update_task(task_id, status="running")
        store.update_model(model_id, status="training")

        # 获取任务信息
        task = store.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # 获取历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # 获取30天历史数据

        metrics = prometheus_api.get_storage_metrics(
            user=task.user, start=start_time, end=end_time, step="1h"
        )

        if "storage_used" not in metrics or metrics["storage_used"].empty:
            raise ValueError("No historical data available")

        # 创建特征矩阵
        features, target = create_feature_matrix(metrics, "storage_used")

        # 创建预测器
        config = ModelConfig(
            input_dim=features.shape[1],
            sequence_length=task.config.get("sequence_length", 24),
            prediction_steps=task.config.get("prediction_steps", 1),
            epochs=task.config.get("epochs", 100),
        )

        predictor = TimeSeriesPredictor(config)

        # 训练模型
        history = predictor.train(features, target)

        # 评估模型
        metrics_result = predictor.evaluate(features, target)

        # 保存模型
        model_name = f"model_{model_id}"
        predictor.save_model(model_name)

        # 更新模型状态
        store.update_model(
            model_id,
            status="completed",
            metrics=metrics_result,
            file_path=f"models/{model_name}.pth",
        )

        # 更新任务状态
        store.update_task(
            task_id,
            status="completed",
            model_id=model_id,
            results={
                "training_history": history,
                "metrics": metrics_result,
                "model_path": f"models/{model_name}.pth",
            },
        )

        logger.info(f"Model training completed for task {task_id}")

    except Exception as e:
        logger.error(f"Model training failed for task {task_id}: {e}")

        # 更新状态为失败
        store.update_task(task_id, status="failed", error_message=str(e))
        store.update_model(model_id, status="failed", error_message=str(e))


# API 路由
@app.get("/")
async def root():
    """根路径"""
    return {"message": "Time Series Forecast Platform API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# 用户管理
@app.post("/users", response_model=UserResponse)
async def create_user(user_data: UserCreate, store: JSONStore = Depends(get_store)):
    """创建用户"""
    try:
        user = store.create_user(user_data.username, user_data.email)
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, store: JSONStore = Depends(get_store)):
    """获取用户信息"""
    user = store.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/users/username/{username}", response_model=UserResponse)
async def get_user_by_username(username: str, store: JSONStore = Depends(get_store)):
    """根据用户名获取用户信息"""
    user = store.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# 任务管理
@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    task_data: TaskCreate,
    background_tasks: BackgroundTasks,
    store: JSONStore = Depends(get_store),
):
    """创建预测任务"""
    try:
        # 创建任务
        task = store.create_task(
            name=task_data.name,
            user=task_data.user,
            metric_query=task_data.metric_query,
            config=task_data.config,
        )

        # 创建模型
        model = store.create_model(
            name=f"Model for {task.name}",
            user=task.user,
            task_id=task.id,
            model_type="lstm",
            config=task_data.config,
        )

        # 更新任务关联模型
        store.update_task(task.id, model_id=model.id)

        # 启动后台训练任务
        background_tasks.add_task(train_model_task, task.id, model.id)

        return task
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, store: JSONStore = Depends(get_store)):
    """获取任务信息"""
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/tasks/user/{username}", response_model=List[TaskResponse])
async def get_user_tasks(username: str, store: JSONStore = Depends(get_store)):
    """获取用户的所有任务"""
    tasks = store.get_tasks_by_user(username)
    return tasks


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str, store: JSONStore = Depends(get_store)):
    """删除任务"""
    success = store.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted successfully"}


# 模型管理
@app.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, store: JSONStore = Depends(get_store)):
    """获取模型信息"""
    model = store.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.get("/models/user/{username}", response_model=List[ModelResponse])
async def get_user_models(username: str, store: JSONStore = Depends(get_store)):
    """获取用户的所有模型"""
    models = store.get_models_by_user(username)
    return models


@app.get("/models/task/{task_id}", response_model=List[ModelResponse])
async def get_task_models(task_id: str, store: JSONStore = Depends(get_store)):
    """获取任务的所有模型"""
    models = store.get_models_by_task(task_id)
    return models


# 预测接口
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    store: JSONStore = Depends(get_store),
    prometheus_api: PrometheusAPI = Depends(get_prometheus_api),
):
    """进行预测"""
    try:
        # 获取模型
        if request.model_id:
            model = store.get_model(request.model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            if model.status != "completed":
                raise HTTPException(
                    status_code=400, detail="Model not ready for prediction"
                )
        else:
            # 获取用户最新的已完成模型
            user_models = store.get_models_by_user(request.user)
            completed_models = [m for m in user_models if m.status == "completed"]
            if not completed_models:
                raise HTTPException(
                    status_code=404, detail="No completed model found for user"
                )

            model = completed_models[-1]  # 取最新的模型

        # 获取历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)  # 获取最近2天数据

        metrics = prometheus_api.get_storage_metrics(
            user=request.user, start=start_time, end=end_time, step="1h"
        )

        if "storage_used" not in metrics or metrics["storage_used"].empty:
            raise HTTPException(status_code=400, detail="No historical data available")

        # 创建特征矩阵
        features, target = create_feature_matrix(metrics, "storage_used")

        # 加载模型
        config = ModelConfig(
            input_dim=features.shape[1],
            sequence_length=model.config.get("sequence_length", 24),
            prediction_steps=request.prediction_steps,
        )

        predictor = TimeSeriesPredictor(config)
        model_name = f"model_{model.id}"
        predictor.load_model(model_name)

        # 进行预测
        predictions = predictor.predict(features)

        # 生成时间戳
        last_timestamp = metrics["storage_used"]["timestamp"].iloc[-1]
        timestamps = []
        for i in range(request.prediction_steps):
            next_time = last_timestamp + timedelta(hours=i + 1)
            timestamps.append(next_time.isoformat())

        return PredictionResponse(
            user=request.user,
            metric_query=request.metric_query,
            predictions=predictions.tolist(),
            timestamps=timestamps,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 数据获取接口
@app.get("/data/metrics/{user}")
async def get_metrics(
    user: str,
    hours: int = 24,
    store: JSONStore = Depends(get_store),
    prometheus_api: PrometheusAPI = Depends(get_prometheus_api),
):
    """获取用户的指标数据"""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        metrics = prometheus_api.get_storage_metrics(
            user=user, start=start_time, end=end_time, step="1h"
        )

        # 转换为前端友好的格式
        result = {}
        for metric_name, df in metrics.items():
            if not df.empty:
                result[metric_name] = {
                    "timestamps": df["timestamp"].dt.isoformat().tolist(),
                    "values": df["value"].tolist(),
                }

        return result

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 统计信息
@app.get("/stats")
async def get_stats(store: JSONStore = Depends(get_store)):
    """获取平台统计信息"""
    return store.get_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
