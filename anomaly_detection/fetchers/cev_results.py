"""
CEV Results data fetcher.
CEV 结果数据获取器
"""

import pandas as pd
from typing import Tuple, List
from ..config.settings import get_db_connection


def fetch_cev_results_data(fleet_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定车队的 CEV 结果比例数据

    统计每天每种事件类型中 cev_result 为 true 占该类型事件的比例。

    Args:
        fleet_id: 车队ID
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        包含 cev_results 比例的数据框
    """
    print(f"[Tool Backend] Fetching CEV results data for fleet: {fleet_id} | {start_date} to {end_date}")
    conn = get_db_connection()

    # 查询每天每种事件类型的 cev_result=true 数量和总数量
    sql_query = """
        SELECT
            DATE(eventtime) as event_date,
            eventtype,
            COUNT(*) as total_events,
            SUM(CASE WHEN cevresult = true THEN 1 ELSE 0 END) as cev_true_events
        FROM v_clip_wide_api
        WHERE fleetid = %s
          AND DATE(eventtime) >= %s
          AND DATE(eventtime) <= %s
          AND eventtype IS NOT NULL
        GROUP BY DATE(eventtime), eventtype
        ORDER BY event_date, eventtype
    """

    try:
        df_raw = pd.read_sql(sql_query, conn, params=(fleet_id, start_date, end_date))
    finally:
        conn.close()

    if df_raw.empty:
        return pd.DataFrame()

    # 计算 cev_result 为 true 的比例
    df_raw['cev_ratio'] = df_raw['cev_true_events'] / df_raw['total_events']

    # 透视数据，将事件类型作为列
    df_pivoted = df_raw.pivot_table(
        index='event_date',
        columns='eventtype',
        values='cev_ratio',
        fill_value=0.0
    ).reset_index()

    # 确保 event_date 是 datetime 类型
    df_pivoted['event_date'] = pd.to_datetime(df_pivoted['event_date'])
    df_pivoted = df_pivoted.sort_values('event_date').reset_index(drop=True)

    # 添加 fleet_id 列
    df_pivoted['fleetid'] = fleet_id

    # 重新排列列顺序
    cols = ['fleetid', 'event_date'] + [c for c in df_pivoted.columns if c not in ['fleetid', 'event_date']]
    df_pivoted = df_pivoted[cols]

    return df_pivoted
