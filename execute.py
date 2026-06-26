#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for running anomaly detection modes one by one.

每种模式单独调用，便于给不同模式配置不同参数；最后把各模式返回的
结构化异常信息统一合并，供后续 LLM 或程序继续消费。
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from anomaly_detection import (
    DEFAULT_TIME_WINDOW_DAYS_EVENT,
    DEFAULT_TIME_WINDOW_DAYS_TRIP_HOUR,
    FleetAnomalyDetectionTool,
    build_time_window_params,
)


STRUCTURED_RESULT_MARKER = "STRUCTURED_ANOMALY_RESULT_JSON:"


def extract_structured_result(report: str) -> Dict[str, Any]:
    """Extract the JSON payload appended by FleetAnomalyDetectionTool._run()."""
    marker_index = report.find(STRUCTURED_RESULT_MARKER)
    if marker_index < 0:
        raise ValueError("工具返回中没有找到结构化 JSON 标记。")

    json_text = report[marker_index + len(STRUCTURED_RESULT_MARKER):].strip()
    return json.loads(json_text)


def build_mode_args(end_date: str = None) -> List[Dict[str, Any]]:
    """Build per-mode arguments. Different modes can tune thresholds separately."""
    fleet_id = "c6329dff7db740a2848b9d34ca6bd7af"
    event_window = build_time_window_params(DEFAULT_TIME_WINDOW_DAYS_EVENT, end_date=end_date)
    events_per_hour_window = build_time_window_params(DEFAULT_TIME_WINDOW_DAYS_TRIP_HOUR, end_date=end_date)

    common_event_args = {
        "fleet_id": fleet_id,
        "start_date": event_window["start_date"],
        "end_date": event_window["end_date"],
        "enable_visualization": False,
        # 趋势回看窗口：每次预测当前日时，只使用当前日之前的历史数据。
        "trend_window": 30,
        # 历史清洗窗口：用于清洗历史异常点，避免尖峰污染后续预测。
        "history_window": 30,
    }

    return [
        {
            **common_event_args,
            "detection_mode": "distribution_shift",
            # 分布漂移灵敏度：越大越不敏感。
            "threshold_multiplier": 8.0,
            # 分布占比变化的最小绝对偏差。
            "min_abs_deviation": 0.25,
            "min_deviation": 0.25,
        },
        {
            **common_event_args,
            "detection_mode": "volume_spike",
            # 强制周期长度。7 表示按星期相位拆分，只用同相位历史预测当天。
            "spike_period": 7,
            # 连续多少个同相位点稳定后，才允许把预测切换到新趋势。
            "stable_regime_points": 3,
            # 稳定新趋势内部允许的相对波动。
            "stable_regime_tolerance_ratio": 0.10,
            # 新趋势相对旧趋势至少变化多少，才认为可能发生台阶切换。
            "stable_shift_min_ratio": 0.45,
            # safe range 相对预测值的基础百分比容忍度。
            "expected_relative_tolerance": 0.9,
            # safe range 最大宽度相对预测值的上限倍数。
            "threshold_width_cap_ratio": 1.0,
            # 低于预测值时的额外放宽倍数，降低低谷误报。
            "lower_anomaly_tolerance_multiplier": 1.0,
            # 历史正常离散度放大倍数。
            "normal_dispersion_multiplier": 3.0,
            # 历史正常离散度的相对下限。
            "normal_dispersion_floor_ratio": 0.25,
            # 至少多少个历史正常点后启用离散度宽度。
            "normal_dispersion_min_points": 4,
            # MAD/尺度阈值倍数，越大越不敏感。
            "threshold_multiplier": 8.0,
            # 事件量指标的最小绝对偏差，避免小数值抖动触发异常。
            "min_abs_deviation": 300.0,
            "min_deviation": 0.25,
        },
        {
            "fleet_id": fleet_id,
            "start_date": events_per_hour_window["start_date"],
            "end_date": events_per_hour_window["end_date"],
            "detection_mode": "events_per_hour",
            "enable_visualization": True,
            # events_per_hour 是连续小数指标，不做星期相位拆分时使用 1。
            "spike_period": 1,
            "stable_regime_points": 3,
            "stable_regime_tolerance_ratio": 0.30,
            "stable_shift_min_ratio": 0.45,
            # 小数指标的相对容忍度。
            "expected_relative_tolerance": 0.20,
            "threshold_width_cap_ratio": 1.0,
            "lower_anomaly_tolerance_multiplier": 1.0,
            "normal_dispersion_multiplier": 1.0,
            "normal_dispersion_floor_ratio": 0.25,
            "normal_dispersion_min_points": 4,
            "threshold_multiplier": 8.0,
            # events_per_hour 量纲很小，所以最小绝对偏差也要小。
            "min_abs_deviation": 0.03,
            "min_deviation": 0.25
        },
        {
            **common_event_args,
            "detection_mode": "cev_results",
            "enable_visualization": True,
            # cev_results 是比例指标，不做星期相位拆分时使用 1。
            "spike_period": 1,
            "stable_regime_points": 3,
            "stable_regime_tolerance_ratio": 0.20,
            "stable_shift_min_ratio": 0.45,
            # 比例指标的相对容忍度。
            "expected_relative_tolerance": 0.30,
            "threshold_width_cap_ratio": 1.0,
            "lower_anomaly_tolerance_multiplier": 1.0,
            "normal_dispersion_multiplier": 1.5,
            "normal_dispersion_floor_ratio": 0.20,
            "normal_dispersion_min_points": 4,
            "threshold_multiplier": 8.0,
            # cev_results 是 0-1 比例，最小绝对偏差设为 0.05。
            "min_abs_deviation": 0.05,
            "min_deviation": 0.10,
        },
    ]


def save_structured_results(results: List[Dict[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
    """Save per-mode structured JSON as a plain list, without any extra wrapping."""
    if not results:
        raise ValueError("没有可汇总的检测结果。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def run_modes_individually(selected_modes: List[str] = None, end_date: str = None) -> List[Dict[str, Any]]:
    """Run multiple modes without using detection_mode='all'."""
    tool = FleetAnomalyDetectionTool()
    mode_args_list = build_mode_args(end_date=end_date)
    if selected_modes:
        selected = {mode.strip().lower() for mode in selected_modes}
        mode_args_list = [
            args for args in mode_args_list
            if args["detection_mode"].lower() in selected
        ]
        if not mode_args_list:
            raise ValueError(f"没有匹配到指定模式: {', '.join(selected_modes)}")

    structured_results = []
    print("=" * 80)
    print("按模式循环运行异常检测，不使用 all 模式")
    print("=" * 80)

    for args in mode_args_list:
        mode = args["detection_mode"]
        print(f"\n运行模式: {mode}")
        print(f"   时间范围: {args['start_date']} 至 {args['end_date']}")
        report = tool._run(**args)
        print(report)
        structured_results.append(extract_structured_result(report))

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    fleet_id = mode_args_list[0]["fleet_id"]
    output_path = Path("agent_workspace") / f"multi_mode_anomaly_results_{fleet_id}_{timestamp}.json"
    results = save_structured_results(structured_results, output_path)

    print("\n" + "=" * 80)
    print("多模式结构化结果列表")
    print("=" * 80)
    print(f"结果文件: {output_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    # 默认运行全部已配置模式，但每个模式都是单独调用，不使用 detection_mode='all'。
    # 也可以指定一个或多个模式：
    #   python execute.py volume_spike events_per_hour
    mode_list = sys.argv[1:] or [
        # "volume_spike",
        # "distribution_shift",
        # "events_per_hour",
        "cev_results",
    ]

    # 结束日期配置：
    # - None: 默认使用今天
    # - "YYYY-MM-DD": 使用指定日期作为所有模式的 end_date，并按各模式窗口天数动态计算 start_date
    END_DATE = None# "2026-04-30"  # 这一天是有异常的，可以供测试使用

    run_modes_individually(mode_list, end_date=END_DATE)
