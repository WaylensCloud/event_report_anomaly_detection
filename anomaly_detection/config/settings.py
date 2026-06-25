"""
Settings and configuration management.
配置和设置管理模块
"""

import os
import json
import warnings
from typing import Dict, Any

# 尝试导入数据库依赖
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    warnings.warn("psycopg2 not found, database functionality will be limited")

# 尝试导入 scipy
try:
    from scipy.spatial.distance import jensenshannon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found, some functionality will be limited")


def load_config(config_path: str = "./config/env.json") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_db_connection(env_path: str = "./config/env.json"):
    """
    获取数据库连接

    Args:
        env_path: 环境配置文件路径

    Returns:
        数据库连接对象
    """
    if not HAS_PSYCOPG2:
        raise RuntimeError("psycopg2 is required for database connections")

    config = load_config(env_path)
    conn_params = config['conn_params']

    return psycopg2.connect(
        host=conn_params['host'],
        database=conn_params['database'],
        user=conn_params['user'],
        password=conn_params['password'],
        port=conn_params['port'],
    )
