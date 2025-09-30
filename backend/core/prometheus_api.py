"""
Prometheus API Data Retrieval Module
Supports pulling historical data from Prometheus for LSTM model training

Author: mmwei3
Email: mmwei3@iflytek.com, 1300042631@qq.com
Date: 2025-08-27
Weather: Cloudy
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrometheusConfig:
    """Prometheus 配置"""

    base_url: str = "http://localhost:9090"
    pushgateway_url: str = "http://localhost:9091"
    timeout: int = 30


class PrometheusAPI:
    """Prometheus API 客户端"""

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout

    def query_range(
        self, query: str, start: datetime, end: datetime, step: str = "1h"
    ) -> pd.DataFrame:
        """
        查询 Prometheus 时间范围数据

        Args:
            query: PromQL 查询语句
            start: 开始时间
            end: 结束时间
            step: 查询步长

        Returns:
            DataFrame with columns: ['timestamp', 'value', 'labels']
        """
        url = f"{self.config.base_url}/api/v1/query_range"

        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data["status"] != "success":
                raise Exception(
                    f"Prometheus query failed: {data.get('error', 'Unknown error')}"
                )

            results = data["data"]["result"]
            if not results:
                logger.warning(f"No data found for query: {query}")
                return pd.DataFrame(columns=["timestamp", "value", "labels"])

            # 处理多个时间序列
            all_data = []
            for result in results:
                metric = result["metric"]
                values = result["values"]

                for timestamp, value in values:
                    all_data.append(
                        {
                            "timestamp": pd.to_datetime(timestamp, unit="s"),
                            "value": float(value),
                            "labels": metric,
                        }
                    )

            df = pd.DataFrame(all_data)
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} data points for query: {query}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query Prometheus: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing Prometheus response: {e}")
            raise

    def get_storage_metrics(
        self, user: str, start: datetime, end: datetime, step: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """
        获取存储相关指标数据

        Args:
            user: 用户名
            start: 开始时间
            end: 结束时间
            step: 查询步长

        Returns:
            包含各种存储指标的字典
        """
        metrics = {}

        # 存储使用量
        storage_used_query = f'storage_used_bytes{{user="{user}"}}'
        metrics["storage_used"] = self.query_range(storage_used_query, start, end, step)

        # 存储配额
        storage_quota_query = f'storage_quota_bytes{{user="{user}"}}'
        metrics["storage_quota"] = self.query_range(
            storage_quota_query, start, end, step
        )

        # 可选：其他相关指标
        try:
            # I/O 吞吐量
            io_query = f'storage_io_throughput{{user="{user}"}}'
            metrics["io_throughput"] = self.query_range(io_query, start, end, step)
        except:
            logger.warning("I/O throughput metric not available")

        try:
            # 活跃用户数
            active_users_query = f'active_users_count{{user="{user}"}}'
            metrics["active_users"] = self.query_range(
                active_users_query, start, end, step
            )
        except:
            logger.warning("Active users metric not available")

        return metrics

    def push_metric(
        self,
        job_name: str,
        metric_name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> bool:
        """
        推送指标到 Prometheus Pushgateway

        Args:
            job_name: 任务名称
            metric_name: 指标名称
            value: 指标值
            labels: 标签字典

        Returns:
            是否推送成功
        """
        if labels is None:
            labels = {}

        # 构建 Prometheus 格式的指标
        label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
        if label_str:
            metric_line = f"{metric_name}{{{label_str}}} {value}\n"
        else:
            metric_line = f"{metric_name} {value}\n"

        url = f"{self.config.pushgateway_url}/metrics/job/{job_name}"

        try:
            response = self.session.post(url, data=metric_line)
            response.raise_for_status()
            logger.info(f"Successfully pushed metric: {metric_name}={value}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to push metric to Pushgateway: {e}")
            return False

    def push_prediction(
        self, user: str, prediction_value: float, prediction_horizon: str = "1h"
    ) -> bool:
        """
        推送预测结果到 Pushgateway

        Args:
            user: 用户名
            prediction_value: 预测值
            prediction_horizon: 预测时间范围

        Returns:
            是否推送成功
        """
        labels = {
            "user": user,
            "prediction_horizon": prediction_horizon,
            "model": "lstm",
        }

        return self.push_metric(
            job_name="storage_predictions",
            metric_name="storage_usage_predicted",
            value=prediction_value,
            labels=labels,
        )


def create_feature_matrix(
    metrics: Dict[str, pd.DataFrame], target_metric: str = "storage_used"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从多个指标创建特征矩阵

    Args:
        metrics: 指标数据字典
        target_metric: 目标指标名称

    Returns:
        (features, target) 特征矩阵和目标值
    """
    if target_metric not in metrics:
        raise ValueError(f"Target metric '{target_metric}' not found in metrics")

    # 以目标指标的时间戳为基准
    base_df = metrics[target_metric].copy()
    base_df = base_df.set_index("timestamp")

    # 合并其他指标
    feature_columns = []
    for metric_name, df in metrics.items():
        if metric_name == target_metric:
            continue

        if not df.empty:
            df_indexed = df.set_index("timestamp")
            # 重采样到相同时间频率
            df_resampled = df_indexed.resample("1H").mean()
            base_df[f"{metric_name}_value"] = df_resampled["value"]
            feature_columns.append(f"{metric_name}_value")

    # 添加时间特征
    base_df["hour"] = base_df.index.hour
    base_df["day_of_week"] = base_df.index.dayofweek
    base_df["day_of_month"] = base_df.index.day

    feature_columns.extend(["hour", "day_of_week", "day_of_month"])

    # 填充缺失值
    base_df = base_df.fillna(method="ffill").fillna(method="bfill")

    # 提取特征和目标
    features = base_df[feature_columns].values
    target = base_df["value"].values

    return features, target


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建 Prometheus API 客户端
    config = PrometheusConfig()
    api = PrometheusAPI(config)

    # 获取最近30天的数据
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)

    try:
        # 获取存储指标
        metrics = api.get_storage_metrics("alice", start_time, end_time)

        # 打印数据概览
        for metric_name, df in metrics.items():
            print(f"\n{metric_name}:")
            print(f"  Data points: {len(df)}")
            if not df.empty:
                print(
                    f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}"
                )
                print(
                    f"  Value range: {df['value'].min():.2f} to {df['value'].max():.2f}"
                )

        # 创建特征矩阵
        if "storage_used" in metrics:
            features, target = create_feature_matrix(metrics)
            print(f"\nFeature matrix shape: {features.shape}")
            print(f"Target shape: {target.shape}")

    except Exception as e:
        print(f"Error: {e}")
