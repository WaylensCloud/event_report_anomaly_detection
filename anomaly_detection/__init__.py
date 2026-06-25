"""
Anomaly Detection Toolkit
时间序列异常检测工具包
"""

__version__ = "1.0.0"
__author__ = "Anomaly Detection Team"

from .tool import FleetAnomalyDetectionTool
from .config.constants import DEFAULT_TIME_WINDOW_DAYS_EVENT, DEFAULT_TIME_WINDOW_DAYS_TRIP_HOUR, build_time_window_params

# 保持向后兼容
DEFAULT_TIME_WINDOW_DAYS = DEFAULT_TIME_WINDOW_DAYS_EVENT

__all__ = [
    "FleetAnomalyDetectionTool",
    "DEFAULT_TIME_WINDOW_DAYS",
    "DEFAULT_TIME_WINDOW_DAYS_EVENT",
    "DEFAULT_TIME_WINDOW_DAYS_TRIP_HOUR",
    "build_time_window_params",
]
