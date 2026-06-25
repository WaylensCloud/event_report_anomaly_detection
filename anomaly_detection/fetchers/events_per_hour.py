"""
Events per hour data fetcher.
每小时事件数数据获取器
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime, timedelta, date
from collections import defaultdict
from ..config.settings import get_db_connection


def _split_time_range_by_day(start_time, end_time):
    """
    将时间范围按天拆分成多个时间段

    Returns:
        [(date, start_datetime, end_datetime, duration_hours), ...]
    """
    if not start_time or not end_time:
        return []

    if start_time > end_time:
        start_time, end_time = end_time, start_time

    result = []
    current_date = start_time.date()
    end_date = end_time.date()

    while current_date <= end_date:
        day_start = datetime.combine(current_date, datetime.min.time())
        day_end = datetime.combine(current_date, datetime.max.time())
        actual_start = max(start_time, day_start)
        actual_end = min(end_time, day_end)
        duration = (actual_end - actual_start).total_seconds() / 3600.0

        if duration > 0:
            result.append((current_date, actual_start, actual_end, duration))

        current_date += timedelta(days=1)

    return result


def fetch_events_per_hour_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 GroundCloud 车队的 events_per_hour 数据

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        包含 events_per_hour 的数据框
    """
    print(f"[Tool Backend] Fetching events_per_hour data: {start_date} to {end_date}")
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT fleetid FROM ods_fleet_info_f
                WHERE name = 'GroundCloud'
                AND dt::text = (SELECT max(dt::text) FROM ods_fleet_info_f)
                LIMIT 1
            """)
            result = cursor.fetchone()
            if not result:
                print("未找到 GroundCloud 车队")
                return pd.DataFrame()
            fleet_id = result[0]
            print(f"GroundCloud fleetid: {fleet_id}")

        # 获取事件数据
        sql_event = """
            SELECT
                v.fleetid,
                v.eventtype,
                date(v.eventtime) as createddate,
                count(*) as count
            FROM
                v_clip_wide_api v
            WHERE
                v.fleetid = %s
                AND date(v.eventtime) >= %s
                AND date(v.eventtime) <= %s
                AND v.eventtype IS NOT NULL
            GROUP BY
                v.fleetid,
                v.eventtype,
                date(v.eventtime)
            ORDER BY
                date(v.eventtime) DESC,
                count(*) DESC
        """
        df_event = pd.read_sql(sql_event, conn, params=(fleet_id, start_date, end_date))

        print(f"Fetched {len(df_event)} event records for fleet {fleet_id} between {start_date} and {end_date}")

        # 计算需要查询的 dt 范围，给前后各留一点缓冲以处理跨天行程
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # 从开始日期前一个月到结束日期后一个月，确保不会漏掉跨天行程
        dt_start = (start_dt - pd.Timedelta(days=30)).strftime('%Y%m')
        dt_end = (end_dt + pd.Timedelta(days=30)).strftime('%Y%m')

        # 获取行程时长原始数据 - 优化查询，只查询需要的时间范围
        sql_duration = """
            SELECT
                t.ignitionstarttime,
                t.ignitionstoptime,
                t.drivingtime,
                t.parkingtime,
                t.createtime::date as createtime_date
            FROM
                ods_original_clip_f t
                JOIN (
                    SELECT
                        ods_camera_info_f.serialnumber
                    FROM
                        ods_camera_info_f
                    WHERE
                        ods_camera_info_f.fleetid = %s
                        AND ods_camera_info_f.dt::text = (
                            SELECT max(ods_camera_info_f_1.dt::text) AS max
                            FROM ods_camera_info_f ods_camera_info_f_1
                        )
                ) camera_info ON t.camerasn::text = camera_info.serialnumber::text
            WHERE
                t.dt::text >= %s
                AND t.dt::text <= %s
        """
        with conn.cursor() as cursor:
            cursor.execute(sql_duration, (fleet_id, dt_start, dt_end))
            raw_duration_results = cursor.fetchall()
            print(f"Fetched {len(raw_duration_results)} duration records for fleet {fleet_id} (dt: {dt_start} to {dt_end})")

    finally:
        conn.close()

    if df_event.empty:
        return pd.DataFrame()

    # 预先计算日期边界，避免在循环中重复计算
    start_date_bound = pd.to_datetime(start_date).date()
    end_date_bound = pd.to_datetime(end_date).date()
    epoch_date = date(1970, 1, 1)

    # 处理时长数据 - 优化版本
    daily_duration = defaultdict(float)

    for row in raw_duration_results:
        ignitionstart, ignitionstop, drivingtime, parkingtime, createtime_date = row

        # 优先使用 ignitionstart/ignitionstop
        start_time = None
        end_time = None

        # 简化属性检查
        if ignitionstart and ignitionstop:
            try:
                if (ignitionstart.date() != epoch_date and
                    ignitionstop.date() != epoch_date):
                    start_time = ignitionstart
                    end_time = ignitionstop
            except (AttributeError, TypeError):
                pass

        # 如果第一组时间无效，尝试使用 drivingtime/parkingtime
        if start_time is None and drivingtime and parkingtime:
            try:
                start_time = drivingtime
                end_time = parkingtime
            except (AttributeError, TypeError):
                continue

        if start_time and end_time:
            # 直接计算时间差
            try:
                if start_time > end_time:
                    start_time, end_time = end_time, start_time

                current_date = start_time.date()
                end_date_range = end_time.date()

                # 直接迭代每一天
                while current_date <= end_date_range:
                    # 只在目标范围内累加
                    if start_date_bound <= current_date <= end_date_bound:
                        day_start = datetime.combine(current_date, datetime.min.time())
                        day_end = datetime.combine(current_date, datetime.max.time())

                        actual_start = max(start_time, day_start)
                        actual_end = min(end_time, day_end)

                        duration = (actual_end - actual_start).total_seconds() / 3600.0
                        if duration > 0:
                            daily_duration[current_date] += duration

                    current_date += timedelta(days=1)
            except (AttributeError, TypeError, ValueError):
                continue

    print(f"Calculated daily durations for {len(daily_duration)} days between {start_date} and {end_date}")

    # 聚合事件数据按天
    df_daily = df_event.groupby(['fleetid', 'createddate']).agg({
        'count': 'sum'
    }).reset_index()

    # 计算 events_per_hour
    df_daily['total_duration_hours'] = df_daily['createddate'].map(lambda d: daily_duration.get(d, 0))
    df_daily['events_per_hour'] = df_daily.apply(
        lambda row: row['count'] / row['total_duration_hours'] if row['total_duration_hours'] > 0 else 0,
        axis=1
    )

    # 重命名列以符合工具接口
    df_daily = df_daily.rename(columns={'createddate': 'event_date'})
    df_daily['event_date'] = pd.to_datetime(df_daily['event_date'])
    df_daily = df_daily.sort_values('event_date').reset_index(drop=True)

    return df_daily
