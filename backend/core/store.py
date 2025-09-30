"""
Simple JSON Storage Module
Used for storing tasks, models, users and other data

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """预测任务"""

    id: str
    name: str
    user: str
    metric_query: str
    status: str  # pending, running, completed, failed
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    model_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class Model:
    """模型信息"""

    id: str
    name: str
    user: str
    task_id: str
    model_type: str  # lstm, arima, etc.
    status: str  # training, completed, failed
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None
    file_path: Optional[str] = None


@dataclass
class User:
    """用户信息"""

    id: str
    username: str
    email: str
    created_at: str
    last_login: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class JSONStore:
    """JSON 文件存储"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # 数据文件路径
        self.tasks_file = self.data_dir / "tasks.json"
        self.models_file = self.data_dir / "models.json"
        self.users_file = self.data_dir / "users.json"

        # 初始化数据文件
        self._init_data_files()

    def _init_data_files(self):
        """初始化数据文件"""
        for file_path in [self.tasks_file, self.models_file, self.users_file]:
            if not file_path.exists():
                with open(file_path, "w") as f:
                    json.dump([], f, indent=2)

    def _load_data(self, file_path: Path) -> List[Dict]:
        """加载数据"""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load data from {file_path}: {e}")
            return []

    def _save_data(self, file_path: Path, data: List[Dict]):
        """保存数据"""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            raise

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return str(uuid.uuid4())

    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        return datetime.now().isoformat()

    # Task 相关方法
    def create_task(
        self, name: str, user: str, metric_query: str, config: Dict[str, Any]
    ) -> Task:
        """创建新任务"""
        task = Task(
            id=self._generate_id(),
            name=name,
            user=user,
            metric_query=metric_query,
            status="pending",
            created_at=self._get_current_time(),
            updated_at=self._get_current_time(),
            config=config,
        )

        tasks = self._load_data(self.tasks_file)
        tasks.append(asdict(task))
        self._save_data(self.tasks_file, tasks)

        logger.info(f"Created task: {task.id}")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        tasks = self._load_data(self.tasks_file)
        for task_data in tasks:
            if task_data["id"] == task_id:
                return Task(**task_data)
        return None

    def get_tasks_by_user(self, user: str) -> List[Task]:
        """获取用户的所有任务"""
        tasks = self._load_data(self.tasks_file)
        user_tasks = []
        for task_data in tasks:
            if task_data["user"] == user:
                user_tasks.append(Task(**task_data))
        return user_tasks

    def update_task(self, task_id: str, **kwargs) -> bool:
        """更新任务"""
        tasks = self._load_data(self.tasks_file)

        for i, task_data in enumerate(tasks):
            if task_data["id"] == task_id:
                # 更新字段
                for key, value in kwargs.items():
                    if hasattr(Task, key):
                        task_data[key] = value

                # 更新修改时间
                task_data["updated_at"] = self._get_current_time()

                self._save_data(self.tasks_file, tasks)
                logger.info(f"Updated task: {task_id}")
                return True

        logger.warning(f"Task not found: {task_id}")
        return False

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        tasks = self._load_data(self.tasks_file)

        for i, task_data in enumerate(tasks):
            if task_data["id"] == task_id:
                del tasks[i]
                self._save_data(self.tasks_file, tasks)
                logger.info(f"Deleted task: {task_id}")
                return True

        logger.warning(f"Task not found: {task_id}")
        return False

    # Model 相关方法
    def create_model(
        self,
        name: str,
        user: str,
        task_id: str,
        model_type: str,
        config: Dict[str, Any],
    ) -> Model:
        """创建新模型"""
        model = Model(
            id=self._generate_id(),
            name=name,
            user=user,
            task_id=task_id,
            model_type=model_type,
            status="training",
            created_at=self._get_current_time(),
            updated_at=self._get_current_time(),
            config=config,
        )

        models = self._load_data(self.models_file)
        models.append(asdict(model))
        self._save_data(self.models_file, models)

        logger.info(f"Created model: {model.id}")
        return model

    def get_model(self, model_id: str) -> Optional[Model]:
        """获取模型"""
        models = self._load_data(self.models_file)
        for model_data in models:
            if model_data["id"] == model_id:
                return Model(**model_data)
        return None

    def get_models_by_user(self, user: str) -> List[Model]:
        """获取用户的所有模型"""
        models = self._load_data(self.models_file)
        user_models = []
        for model_data in models:
            if model_data["user"] == user:
                user_models.append(Model(**model_data))
        return user_models

    def get_models_by_task(self, task_id: str) -> List[Model]:
        """获取任务的所有模型"""
        models = self._load_data(self.models_file)
        task_models = []
        for model_data in models:
            if model_data["task_id"] == task_id:
                task_models.append(Model(**model_data))
        return task_models

    def update_model(self, model_id: str, **kwargs) -> bool:
        """更新模型"""
        models = self._load_data(self.models_file)

        for i, model_data in enumerate(models):
            if model_data["id"] == model_id:
                # 更新字段
                for key, value in kwargs.items():
                    if hasattr(Model, key):
                        model_data[key] = value

                # 更新修改时间
                model_data["updated_at"] = self._get_current_time()

                self._save_data(self.models_file, models)
                logger.info(f"Updated model: {model_id}")
                return True

        logger.warning(f"Model not found: {model_id}")
        return False

    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        models = self._load_data(self.models_file)

        for i, model_data in enumerate(models):
            if model_data["id"] == model_id:
                del models[i]
                self._save_data(self.models_file, models)
                logger.info(f"Deleted model: {model_id}")
                return True

        logger.warning(f"Model not found: {model_id}")
        return False

    # User 相关方法
    def create_user(self, username: str, email: str) -> User:
        """创建新用户"""
        user = User(
            id=self._generate_id(),
            username=username,
            email=email,
            created_at=self._get_current_time(),
            preferences={},
        )

        users = self._load_data(self.users_file)
        users.append(asdict(user))
        self._save_data(self.users_file, users)

        logger.info(f"Created user: {user.id}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        users = self._load_data(self.users_file)
        for user_data in users:
            if user_data["id"] == user_id:
                return User(**user_data)
        return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        users = self._load_data(self.users_file)
        for user_data in users:
            if user_data["username"] == username:
                return User(**user_data)
        return None

    def update_user(self, user_id: str, **kwargs) -> bool:
        """更新用户"""
        users = self._load_data(self.users_file)

        for i, user_data in enumerate(users):
            if user_data["id"] == user_id:
                # 更新字段
                for key, value in kwargs.items():
                    if hasattr(User, key):
                        user_data[key] = value

                self._save_data(self.users_file, users)
                logger.info(f"Updated user: {user_id}")
                return True

        logger.warning(f"User not found: {user_id}")
        return False

    # 统计方法
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        tasks = self._load_data(self.tasks_file)
        models = self._load_data(self.models_file)
        users = self._load_data(self.users_file)

        # 任务统计
        task_stats = {
            "total": len(tasks),
            "pending": len([t for t in tasks if t["status"] == "pending"]),
            "running": len([t for t in tasks if t["status"] == "running"]),
            "completed": len([t for t in tasks if t["status"] == "completed"]),
            "failed": len([t for t in tasks if t["status"] == "failed"]),
        }

        # 模型统计
        model_stats = {
            "total": len(models),
            "training": len([m for m in models if m["status"] == "training"]),
            "completed": len([m for m in models if m["status"] == "completed"]),
            "failed": len([m for m in models if m["status"] == "failed"]),
        }

        return {"tasks": task_stats, "models": model_stats, "users": len(users)}


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建存储实例
    store = JSONStore()

    # 创建用户
    user = store.create_user("alice", "alice@example.com")
    print(f"Created user: {user.username}")

    # 创建任务
    task_config = {"sequence_length": 24, "prediction_steps": 1, "epochs": 100}

    task = store.create_task(
        name="Storage Usage Prediction",
        user=user.username,
        metric_query="storage_used_bytes{user='alice'}",
        config=task_config,
    )
    print(f"Created task: {task.name}")

    # 创建模型
    model_config = {"input_dim": 3, "hidden_dim": 64, "num_layers": 2}

    model = store.create_model(
        name="LSTM Model v1",
        user=user.username,
        task_id=task.id,
        model_type="lstm",
        config=model_config,
    )
    print(f"Created model: {model.name}")

    # 更新任务状态
    store.update_task(task.id, status="running", model_id=model.id)

    # 更新模型状态和指标
    metrics = {"mse": 0.001, "mae": 0.02, "rmse": 0.03, "mape": 5.2}
    store.update_model(
        model.id,
        status="completed",
        metrics=metrics,
        file_path="models/lstm_model_20231201_120000.pth",
    )

    # 获取统计信息
    stats = store.get_stats()
    print(f"Statistics: {stats}")

    # 获取用户的任务
    user_tasks = store.get_tasks_by_user(user.username)
    print(f"User tasks: {len(user_tasks)}")

    # 获取任务的模型
    task_models = store.get_models_by_task(task.id)
    print(f"Task models: {len(task_models)}")
