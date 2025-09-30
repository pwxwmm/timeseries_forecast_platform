"""
Core Module Package
Contains core application components including FastAPI app, storage, and Prometheus API

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

from .app import app
from .store import JSONStore, Task, Model, User
from .forecast import TimeSeriesPredictor, ModelConfig
from .prometheus_api import PrometheusAPI, PrometheusConfig

__all__ = [
    "app",
    "JSONStore",
    "Task",
    "Model",
    "User",
    "TimeSeriesPredictor",
    "ModelConfig",
    "PrometheusAPI",
    "PrometheusConfig",
]
