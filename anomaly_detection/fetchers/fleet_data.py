"""
Fleet data fetcher.
车队数据获取器
"""

import pandas as pd
from typing import Tuple, List
from ..config.settings import get_db_connection


def fetch_fleet_data(fleet_id: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    获取车队事件数据

    Args:
        fleet_id: 车队ID
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        (数据框, 目标列列表)
    """
    print(f"[Tool Backend] Fetching DB for fleet: {fleet_id} | {start_date} to {end_date}")
    conn = get_db_connection()

    sql_query = f"""
        SELECT fleetid, DATE(eventtime) as event_date, eventtype, COUNT(*) as event_count
        FROM v_clip_wide_api
        WHERE fleetid = '{fleet_id}' AND DATE(eventtime) >= '{start_date}' AND DATE(eventtime) <= '{end_date}'
        GROUP BY fleetid, DATE(eventtime), eventtype
    """
    try:
        df_raw = pd.read_sql(sql_query, conn)
    finally:
        conn.close()

    if df_raw.empty:
        return pd.DataFrame(), []

    df_pivoted = df_raw.pivot_table(index=['fleetid', 'event_date'], columns='eventtype', values='event_count', fill_value=0).reset_index()
    df_pivoted['event_date'] = pd.to_datetime(df_pivoted['event_date'])
    df_pivoted = df_pivoted.sort_values('event_date').reset_index(drop=True)
    target_columns = [col for col in df_pivoted.columns if col not in ['fleetid', 'event_date']]
    return df_pivoted, target_columns
