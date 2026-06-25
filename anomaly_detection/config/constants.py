"""
Constants and utility functions for anomaly detection.
异常检测常量和工具函数
"""

from typing import Any, Dict
import pandas as pd
# from datetime import datetime


DEFAULT_TIME_WINDOW_DAYS_TRIP_HOUR = 30
DEFAULT_TIME_WINDOW_DAYS_EVENT = 90
DEFAULT_TIME_WINDOW_DAYS = DEFAULT_TIME_WINDOW_DAYS_EVENT


def _today_str() -> str:
    """获取今天的日期字符串"""
    return pd.Timestamp.today().strftime('%Y-%m-%d')


def _date_days_ago_str(time_window_days: int, end_date: str = None) -> str:
    """获取N天前的日期字符串"""
    end = pd.to_datetime(end_date or _today_str()).normalize()
    return (end - pd.Timedelta(days=int(time_window_days))).strftime('%Y-%m-%d')


def build_time_window_params(time_window_days: int = DEFAULT_TIME_WINDOW_DAYS_EVENT, end_date: str = None) -> Dict[str, Any]:
    """构建时间窗口参数字典"""
    end_date = end_date or _today_str()
    return {
        "time_window_days": time_window_days,
        "start_date": _date_days_ago_str(time_window_days, end_date),
        "end_date": end_date,
    }
