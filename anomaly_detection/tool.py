"""
Main anomaly detection tool.
主异常检测工具类
"""

import os
import json
import pandas as pd
from typing import Any, Dict, Optional, Type, List, Tuple
from datetime import datetime
import warnings

# 导入 LangChain 工具定义的标准库
try:
    from langchain_core.tools import BaseTool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    class BaseTool: pass

# 导入内部模块
# from .config.constants import DEFAULT_TIME_WINDOW_DAYS
from .core.engine_clean import AnomalyDetectorEngine
from .fetchers import fetch_fleet_data, fetch_events_per_hour_data

warnings.filterwarnings("ignore")


class FleetAnomalyDetectionTool(BaseTool):
    """
    车队行为异常检测工具

    支持的检测模式：
    - distribution_shift: 分布占比漂移崩塌
    - volume_spike: 总量和单项突增
    - events_per_hour: 每小时事件数异常
    - all: 运行所有模式
    """
    name: str = "fleet_behavior_anomaly_detector_periodic_regime_variance"
    description: str = (
        "扫描指定时间段的车队告警数据，提供每日维度的宏观异常检验并生成可视化图表。"
        "模式支持：结构占比漂移崩塌 'distribution_shift'，按当前趋势历史均值百分比范围判定的 'volume_spike'，"
        "以及每小时事件数异常检测 'events_per_hour'。"
    )

    env_path: str = "./config/env.json"

    @staticmethod
    def _clean_number(value: Any):
        """
        清理数值，处理NaN

        Args:
            value: 输入值

        Returns:
            清理后的值
        """
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _build_structured_result(
        fleet_id: str, start_date: str, end_date: str, last_event_date: pd.Timestamp,
        modes: List[str], total_days: int, anomaly_records: List[Dict[str, Any]],
        summary_path: str
    ) -> Dict[str, Any]:
        """
        构建结构化结果

        Args:
            fleet_id: 车队ID
            start_date: 开始日期
            end_date: 结束日期
            last_event_date: 最后事件日期
            modes: 检测模式
            total_days: 总天数
            anomaly_records: 异常记录
            summary_path: 摘要路径

        Returns:
            结构化结果字典
        """
        return {
            "summary": {
                "fleet_id": fleet_id,
                "event_window": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "judgment_date": last_event_date.strftime('%Y-%m-%d')
                },
                "detection_modes": modes,
                "total_days": total_days,
                "has_anomaly": len(anomaly_records) > 0,
                "anomaly_count": len(anomaly_records),
                "summary_file": summary_path
            },
            "anomalies": anomaly_records
        }

    @classmethod
    def _build_anomaly_record(
        cls, fleet_id: str, start_date: str, end_date: str, event_date: pd.Timestamp,
        detection_mode: str, eventtype: str, metric: str, actual_value: Any,
        expected_value: Any, threshold_lower: Any, threshold_upper: Any,
        anomaly_score: Any, reason: str, extra: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        构建单条异常记录

        Args:
            fleet_id: 车队ID
            start_date: 开始日期
            end_date: 结束日期
            event_date: 事件日期
            detection_mode: 检测模式
            eventtype: 事件类型
            metric: 指标
            actual_value: 实际值
            expected_value: 预期值
            threshold_lower: 阈值下界
            threshold_upper: 阈值上界
            anomaly_score: 异常分数
            reason: 原因
            extra: 额外信息

        Returns:
            异常记录字典
        """
        actual = cls._clean_number(actual_value)
        expected = cls._clean_number(expected_value)
        lower = cls._clean_number(threshold_lower)
        upper = cls._clean_number(threshold_upper)
        score = cls._clean_number(anomaly_score)

        if actual is None or expected is None:
            direction = "unknown"
        elif actual > expected:
            direction = "too_high"
        elif actual < expected:
            direction = "too_low"
        else:
            direction = "equal_to_expected"

        record = {
            "fleet_id": fleet_id,
            "event_window": {
                "start_date": start_date,
                "end_date": end_date,
                "judgment_date": event_date.strftime('%Y-%m-%d')
            },
            "detection_mode": detection_mode,
            "eventtype": eventtype,
            "metric": metric,
            "actual_value": actual,
            "expected_value": expected,
            "threshold_interval": {
                "lower": lower,
                "upper": upper
            },
            "direction": direction,
            "is_anomaly": True,
            "anomaly_score": score,
            "reason": reason
        }
        if extra:
            record["extra"] = extra
        return record

    @classmethod
    def _collect_last_day_anomaly_records(
        cls, mode_results: List[Tuple[str, pd.DataFrame, Optional[str], str]],
        target_columns: List[str], fleet_id: str, start_date: str,
        end_date: str, last_event_date: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """
        收集最后一天的异常记录

        Args:
            mode_results: 模式结果列表
            target_columns: 目标列
            fleet_id: 车队ID
            start_date: 开始日期
            end_date: 结束日期
            last_event_date: 最后事件日期

        Returns:
            异常记录列表
        """
        records = []

        for mode, res_df, _, _ in mode_results:
            last_row = res_df.iloc[-1]
            if not bool(last_row['is_anomaly']):
                continue

            if mode == "volume_spike":
                if bool(last_row.get('is_total_anomaly', False)):
                    expected = last_row['expected_total']
                    upper = last_row['total_upper_bound']
                    lower = last_row.get('total_lower_bound', max(0.0, expected - (upper - expected)))
                    records.append(cls._build_anomaly_record(
                        fleet_id, start_date, end_date, last_event_date,
                        mode, "__total__", "event_count", last_row['total'],
                        expected, lower, upper, last_row['anomaly_score'],
                        str(last_row.get('anomaly_reason', '总量异常')),
                        {
                            "scope": "total_volume",
                            "model": "periodic_phase_regime_variance_robust",
                            "spike_period": cls._clean_number(last_row.get('spike_period', None)),
                            "phase": int((len(res_df) - 1) % int(last_row.get('spike_period', 7))) + 1,
                            "trained_inlier_count": cls._clean_number(last_row.get('total_ml_train_inliers', None)),
                            "regime_shift_active": bool(last_row.get('total_regime_shift_active', False)),
                            "regime_level": cls._clean_number(last_row.get('total_regime_level', None))
                        }
                    ))

                for col in target_columns:
                    anom_col = f'sub_anom_{col}'
                    if anom_col not in last_row.index or not bool(last_row[anom_col]):
                        continue

                    expected = last_row[f'sub_exp_{col}']
                    upper = last_row[f'sub_ub_{col}']
                    lower = last_row.get(f'sub_lb_{col}', max(0.0, expected - (upper - expected)))
                    score_col = f'sub_score_{col}'
                    records.append(cls._build_anomaly_record(
                        fleet_id, start_date, end_date, last_event_date,
                        mode, col, "event_count", last_row[col],
                        expected, lower, upper, last_row.get(score_col, last_row['anomaly_score']),
                        f"{col} 计数异常",
                        {
                            "scope": "single_eventtype",
                            "model": "periodic_phase_regime_variance_robust",
                            "spike_period": cls._clean_number(last_row.get('spike_period', None)),
                            "phase": int((len(res_df) - 1) % int(last_row.get('spike_period', 7))) + 1,
                            "trained_inlier_count": cls._clean_number(last_row.get(f'sub_ml_train_inliers_{col}', None)),
                            "regime_shift_active": bool(last_row.get(f'sub_regime_shift_active_{col}', False)),
                            "regime_level": cls._clean_number(last_row.get(f'sub_regime_level_{col}', None))
                        }
                    ))

            elif mode == "distribution_shift":
                threshold_lower = 0.0
                threshold_upper = last_row['upper_bound']
                contributors = []
                for col in target_columns:
                    actual_prop_col = f'{col}_prop'
                    expected_prop_col = f'{col}_expected_prop'
                    if actual_prop_col not in last_row.index or expected_prop_col not in last_row.index:
                        continue
                    actual_prop = last_row[actual_prop_col]
                    expected_prop = last_row[expected_prop_col]
                    contributors.append((col, abs(actual_prop - expected_prop), actual_prop, expected_prop))

                contributors = [item for item in contributors if item[1] > 1e-9]
                contributors.sort(key=lambda item: item[1], reverse=True)
                for col, delta, actual_prop, expected_prop in contributors:
                    records.append(cls._build_anomaly_record(
                        fleet_id, start_date, end_date, last_event_date,
                        mode, col, "event_proportion", actual_prop,
                        expected_prop, threshold_lower, threshold_upper,
                        last_row['anomaly_score'], str(last_row.get('anomaly_reason', '分布漂移异常')),
                        {
                            "js_divergence": cls._clean_number(last_row['js_divergence']),
                            "baseline_divergence": cls._clean_number(last_row['baseline_divergence']),
                            "proportion_delta": cls._clean_number(actual_prop - expected_prop),
                            "threshold_metric": "js_divergence",
                            "actual_count": cls._clean_number(last_row[col]),
                            "expected_count": cls._clean_number(expected_prop * last_row['total']),
                            "total_count": cls._clean_number(last_row['total'])
                        }
                    ))

            elif mode == "events_per_hour":
                expected = last_row['expected_events_per_hour']
                upper = last_row['events_per_hour_upper_bound']
                lower = last_row.get('events_per_hour_lower_bound', max(0.0, expected - (upper - expected)))
                records.append(cls._build_anomaly_record(
                    fleet_id, start_date, end_date, last_event_date,
                    mode, "events_per_hour", "events_per_hour", last_row['events_per_hour'],
                    expected, lower, upper, last_row['anomaly_score'],
                    str(last_row.get('anomaly_reason', '每小时事件数异常')),
                    {
                        "scope": "events_per_hour",
                        "model": "periodic_phase_regime_variance_robust",
                        "spike_period": cls._clean_number(last_row.get('spike_period', None)),
                        "phase": int((len(res_df) - 1) % int(last_row.get('spike_period', 7))) + 1,
                        "trained_inlier_count": cls._clean_number(last_row.get('events_per_hour_ml_train_inliers', None)),
                        "regime_shift_active": bool(last_row.get('events_per_hour_regime_shift_active', False)),
                        "regime_level": cls._clean_number(last_row.get('events_per_hour_regime_level', None)),
                        "total_count": cls._clean_number(last_row.get('count', None)),
                        "total_duration_hours": cls._clean_number(last_row.get('total_duration_hours', None))
                    }
                ))

        return records

    @staticmethod
    def _resolve_detection_modes(detection_mode: str) -> List[str]:
        """
        解析检测模式

        Args:
            detection_mode: 检测模式字符串

        Returns:
            模式列表
        """
        mode = (detection_mode or "all").strip().lower()
        if mode in {"all", "both", "全部", "所有"}:
            return ["distribution_shift", "volume_spike", "events_per_hour"]
        if mode in {"distribution_shift", "volume_spike", "events_per_hour"}:
            return [mode]
        return []

    @staticmethod
    def _run_one_mode(
        mode: str, df: pd.DataFrame, target_columns: List[str], fleet_id: str,
        timestamp: str, out_dir: str, threshold_multiplier: float,
        min_abs_deviation: float, min_deviation: float, trend_window: int,
        history_window: int, enable_visualization: bool, spike_period: int = 7,
        stable_regime_points: int = 3,
        stable_regime_tolerance_ratio: float = 0.10,
        stable_shift_min_ratio: float = 0.45,
        expected_relative_tolerance: float = 0.65,
        threshold_width_cap_ratio: float = 1.80,
        lower_anomaly_tolerance_multiplier: float = 1.80,
        normal_dispersion_multiplier: float = 3.0,
        normal_dispersion_floor_ratio: float = 0.25,
        normal_dispersion_min_points: int = 4
    ) -> Tuple[pd.DataFrame, Optional[str], str]:
        """
        运行单个检测模式

        Args:
            mode: 检测模式
            df: 数据框
            target_columns: 目标列
            fleet_id: 车队ID
            timestamp: 时间戳
            out_dir: 输出目录
            threshold_multiplier: 阈值乘数
            min_abs_deviation: 最小绝对偏差
            min_deviation: 最小偏差
            trend_window: 趋势窗口
            history_window: 历史窗口
            enable_visualization: 是否启用可视化
            spike_period: 周期长度
            stable_regime_points: 稳定区间点数
            stable_regime_tolerance_ratio: 稳定区间容差
            stable_shift_min_ratio: 稳定切换最小比例
            expected_relative_tolerance: 预测值附近的最小相对容忍区间
            threshold_width_cap_ratio: 阈值区间最大可扩到 expected 的比例
            lower_anomaly_tolerance_multiplier: 低于预测值时额外放宽倍数
            normal_dispersion_multiplier: 历史正常离散度放大倍数
            normal_dispersion_floor_ratio: 历史离散度阈值相对下限
            normal_dispersion_min_points: 启用历史离散度判断的最少已确认点数

        Returns:
            (结果数据框, CSV路径, 图表路径)
        """
        if mode == "distribution_shift":
            res_df = AnomalyDetectorEngine.run_distribution_shift_sliding(
                df, target_columns, threshold_multiplier, min_deviation, trend_window, history_window
            )
            plot_path = f"{out_dir}/distribution_curve_{fleet_id}_{timestamp}.png"
            if enable_visualization:
                AnomalyDetectorEngine.plot_distribution_shift_sliding(res_df, target_columns, fleet_id, plot_path)
        elif mode == "volume_spike":
            res_df = AnomalyDetectorEngine.run_volume_spike_sliding(
                df, target_columns, threshold_multiplier, min_abs_deviation, trend_window, history_window,
                spike_period, stable_regime_points, stable_regime_tolerance_ratio, stable_shift_min_ratio,
                expected_relative_tolerance, threshold_width_cap_ratio,
                lower_anomaly_tolerance_multiplier, normal_dispersion_multiplier,
                normal_dispersion_floor_ratio, normal_dispersion_min_points
            )
            plot_path = f"{out_dir}/volume_curve_{fleet_id}_{timestamp}.png"
            if enable_visualization:
                AnomalyDetectorEngine.plot_volume_spike_sliding(res_df, fleet_id, plot_path, spike_period)
        else:
            raise ValueError(f"Unsupported detection mode: {mode}")

        csv_path = None
        if enable_visualization:
            csv_path = f"{out_dir}/scan_data_{mode}_{fleet_id}_{timestamp}.csv"
            res_df.to_csv(csv_path, index=False)
        return res_df, csv_path, plot_path

    @staticmethod
    def _run_events_per_hour_mode(
        df: pd.DataFrame, fleet_id: str, timestamp: str, out_dir: str,
        threshold_multiplier: float, min_abs_deviation: float, min_deviation: float,
        trend_window: int, history_window: int, enable_visualization: bool,
        spike_period: int = 7, stable_regime_points: int = 3,
        stable_regime_tolerance_ratio: float = 0.10,
        stable_shift_min_ratio: float = 0.45,
        expected_relative_tolerance: float = 0.65,
        threshold_width_cap_ratio: float = 1.80,
        lower_anomaly_tolerance_multiplier: float = 1.80,
        normal_dispersion_multiplier: float = 3.0,
        normal_dispersion_floor_ratio: float = 0.25,
        normal_dispersion_min_points: int = 4
    ) -> Tuple[pd.DataFrame, Optional[str], str]:
        """
        运行 events_per_hour 检测模式

        Args:
            df: 数据框
            fleet_id: 车队ID
            timestamp: 时间戳
            out_dir: 输出目录
            threshold_multiplier: 阈值乘数
            min_abs_deviation: 最小绝对偏差
            min_deviation: 最小偏差
            trend_window: 趋势窗口
            history_window: 历史窗口
            enable_visualization: 是否启用可视化
            spike_period: 周期长度
            stable_regime_points: 稳定区间点数
            stable_regime_tolerance_ratio: 稳定区间容差
            stable_shift_min_ratio: 稳定切换最小比例
            expected_relative_tolerance: 预测值附近的最小相对容忍区间
            threshold_width_cap_ratio: 阈值区间最大可扩到 expected 的比例
            lower_anomaly_tolerance_multiplier: 低于预测值时额外放宽倍数
            normal_dispersion_multiplier: 历史正常离散度放大倍数
            normal_dispersion_floor_ratio: 历史离散度阈值相对下限
            normal_dispersion_min_points: 启用历史离散度判断的最少已确认点数

        Returns:
            (结果数据框, CSV路径, 图表路径)
        """
        res_df = AnomalyDetectorEngine.run_events_per_hour_spike(
            df, threshold_multiplier, min_abs_deviation, trend_window, history_window,
            spike_period, stable_regime_points, stable_regime_tolerance_ratio, stable_shift_min_ratio,
            expected_relative_tolerance, threshold_width_cap_ratio,
            lower_anomaly_tolerance_multiplier, normal_dispersion_multiplier,
            normal_dispersion_floor_ratio, normal_dispersion_min_points
        )
        plot_path = f"{out_dir}/events_per_hour_curve_{fleet_id}_{timestamp}.png"
        if enable_visualization:
            AnomalyDetectorEngine.plot_events_per_hour_spike(res_df, fleet_id, plot_path, spike_period)

        csv_path = None
        if enable_visualization:
            csv_path = f"{out_dir}/scan_data_events_per_hour_{fleet_id}_{timestamp}.csv"
            res_df.to_csv(csv_path, index=False)
        return res_df, csv_path, plot_path

    def _mock_run_for_testing(
        self, fleet_id, start_date, end_date, detection_mode, enable_visualization,
        th_mult, min_abs_dev, min_dev, trend_win, hist_win, spike_period=7,
        stable_regime_points=3, stable_regime_tolerance_ratio=0.10,
        stable_shift_min_ratio=0.45, expected_relative_tolerance=0.65,
        threshold_width_cap_ratio=1.80, lower_anomaly_tolerance_multiplier=1.80,
        normal_dispersion_multiplier=3.0, normal_dispersion_floor_ratio=0.25,
        normal_dispersion_min_points=4
    ):
        """
        使用本地CSV文件的Mock运行

        Args:
            fleet_id: 车队ID
            start_date: 开始日期
            end_date: 结束日期
            detection_mode: 检测模式
            enable_visualization: 是否启用可视化
            th_mult: 阈值乘数
            min_abs_dev: 最小绝对偏差
            min_dev: 最小偏差
            trend_win: 趋势窗口
            hist_win: 历史窗口
            spike_period: 周期长度
            stable_regime_points: 稳定区间点数
            stable_regime_tolerance_ratio: 稳定区间容差
            stable_shift_min_ratio: 稳定切换最小比例
            expected_relative_tolerance: 预测值附近的最小相对容忍区间
            threshold_width_cap_ratio: 阈值区间最大可扩到 expected 的比例
            lower_anomaly_tolerance_multiplier: 低于预测值时额外放宽倍数
            normal_dispersion_multiplier: 历史正常离散度放大倍数
            normal_dispersion_floor_ratio: 历史离散度阈值相对下限
            normal_dispersion_min_points: 启用历史离散度判断的最少已确认点数

        Returns:
            结果字符串
        """
        modes = self._resolve_detection_modes(detection_mode)
        if not modes:
            return f"⚠️ 未知的检测模式: {detection_mode}。仅支持 'all'、'distribution_shift'、'volume_spike'、'events_per_hour'。"

        out_dir = './agent_workspace'
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M')

        mode_results = []
        target_columns = []

        # 处理需要原始事件数据的模式
        event_modes = [m for m in modes if m in ['distribution_shift', 'volume_spike']]
        if event_modes:
            df = pd.read_csv('./data/all_fleets_timeseries.csv')
            df['event_date'] = pd.to_datetime(df['event_date'])
            df = df[df['fleetid'] == fleet_id]
            mask = (df['event_date'] >= pd.to_datetime(start_date)) & (df['event_date'] <= pd.to_datetime(end_date))
            df = df.loc[mask].sort_values('event_date').reset_index(drop=True)
            target_columns = [col for col in df.columns if col not in ['fleetid', 'event_date', 'fleetname', 'total_events']]

            if not df.empty and target_columns:
                for mode in event_modes:
                    res_df, csv_path, plot_path = self._run_one_mode(
                        mode, df, target_columns, fleet_id, f"mock_{timestamp}", out_dir,
                        th_mult, min_abs_dev, min_dev, trend_win, hist_win, enable_visualization, spike_period,
                        stable_regime_points, stable_regime_tolerance_ratio, stable_shift_min_ratio,
                        expected_relative_tolerance, threshold_width_cap_ratio,
                        lower_anomaly_tolerance_multiplier, normal_dispersion_multiplier,
                        normal_dispersion_floor_ratio, normal_dispersion_min_points
                    )
                    mode_results.append((mode, res_df, csv_path, plot_path))

        # 处理 events_per_hour 模式（mock数据）
        if 'events_per_hour' in modes:
            mock_eph_path = '/home/yongliu/workspace/redshift/groundcloud_event_stats.csv'
            if os.path.exists(mock_eph_path):
                df_eph = pd.read_csv(mock_eph_path)
                df_eph['event_date'] = pd.to_datetime(df_eph['date'])
                mask = (df_eph['event_date'] >= pd.to_datetime(start_date)) & (df_eph['event_date'] <= pd.to_datetime(end_date))
                df_eph = df_eph.loc[mask].sort_values('event_date').reset_index(drop=True)
                df_eph['fleetid'] = fleet_id

                if not df_eph.empty:
                    res_df, csv_path, plot_path = self._run_events_per_hour_mode(
                        df_eph, "GroundCloud", f"mock_{timestamp}", out_dir,
                        th_mult, min_abs_dev, min_dev, trend_win, hist_win, enable_visualization, spike_period,
                        stable_regime_points, stable_regime_tolerance_ratio, stable_shift_min_ratio,
                        expected_relative_tolerance, threshold_width_cap_ratio,
                        lower_anomaly_tolerance_multiplier, normal_dispersion_multiplier,
                        normal_dispersion_floor_ratio, normal_dispersion_min_points
                    )
                    mode_results.append(('events_per_hour', res_df, csv_path, plot_path))

        if not mode_results:
            return f"⚠️ MOCK 无法检测：指定时间段内没有获取到有效数据。"

        last_event_date = mode_results[0][1]['event_date'].iloc[-1]
        last_day_anomalies = [
            (mode, res_df.iloc[-1])
            for mode, res_df, _, _ in mode_results
            if bool(res_df.iloc[-1]['is_anomaly'])
        ]
        anomaly_records = self._collect_last_day_anomaly_records(
            mode_results, target_columns, fleet_id, start_date, end_date, last_event_date
        )
        summary_path = f"{out_dir}/last_day_anomaly_summary_{fleet_id}_mock_{timestamp}.json"
        structured_result = self._build_structured_result(
            fleet_id, start_date, end_date, last_event_date, modes,
            len(mode_results[0][1]), anomaly_records, summary_path
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(structured_result, f, ensure_ascii=False, indent=2)

        report = [
            f"[MOCK 事件窗口末日异常检测完成]",
            f"从 {start_date} 到 {end_date}，判定日期 {last_event_date.strftime('%Y-%m-%d')}。",
            f"最后一天共被 {len(last_day_anomalies)} 个模式判定为异常，汇总到 {len(anomaly_records)} 条结构化异常信息。"
        ]
        for mode, row in sorted(last_day_anomalies, key=lambda item: item[1]['anomaly_score'], reverse=True):
            report.append(f"  - 模式: {mode} | 严重度评分: {row['anomaly_score']:.2f} | 原因: {row['anomaly_reason']}")
        report.append("详细评估底表:")
        if enable_visualization:
            for mode, _, csv_path, plot_path in mode_results:
                report.append(f"  - {mode}: `{csv_path}`")
                report.append(f"    图表: `{plot_path}`")
        else:
            report.append("  - 已关闭：enable_visualization=False 时不保存 CSV。")
        report.append(f"末日异常汇总文件: `{summary_path}`")
        report.append("\nSTRUCTURED_ANOMALY_RESULT_JSON:")
        report.append(json.dumps(structured_result, ensure_ascii=False, indent=2))
        return "\n".join(report)

    def _run(
        self, fleet_id: str, start_date: str, end_date: str,
        detection_mode: str = "all",
        enable_visualization: bool = False,
        spike_period: int = 7,
        stable_regime_points: int = 3,
        stable_regime_tolerance_ratio: float = 0.10,
        stable_shift_min_ratio: float = 0.45,
        expected_relative_tolerance: float = 0.65,
        threshold_width_cap_ratio: float = 1.80,
        lower_anomaly_tolerance_multiplier: float = 1.80,
        normal_dispersion_multiplier: float = 3.0,
        normal_dispersion_floor_ratio: float = 0.25,
        normal_dispersion_min_points: int = 4,
        threshold_multiplier: float = 8.0,
        min_abs_deviation: float = 120.0,
        min_deviation: float = 0.25,
        trend_window: int = 21,
        history_window: int = 30
    ) -> str:
        """
        运行异常检测

        Args:
            fleet_id: 车队ID
            start_date: 开始日期
            end_date: 结束日期
            detection_mode: 检测模式
            enable_visualization: 是否启用可视化
            spike_period: 周期长度
            stable_regime_points: 稳定区间点数
            stable_regime_tolerance_ratio: 稳定区间容差
            stable_shift_min_ratio: 稳定切换最小比例
            expected_relative_tolerance: 预测值附近的最小相对容忍区间
            threshold_width_cap_ratio: 阈值区间最大可扩到 expected 的比例
            lower_anomaly_tolerance_multiplier: 低于预测值时额外放宽倍数
            normal_dispersion_multiplier: 历史正常离散度放大倍数
            normal_dispersion_floor_ratio: 历史离散度阈值相对下限
            normal_dispersion_min_points: 启用历史离散度判断的最少已确认点数
            threshold_multiplier: 阈值乘数
            min_abs_deviation: 最小绝对偏差
            min_deviation: 最小偏差
            trend_window: 趋势窗口
            history_window: 历史窗口

        Returns:
            结果字符串
        """
        try:
            modes = self._resolve_detection_modes(detection_mode)
            if not modes:
                return f"⚠️ 未知的检测模式: {detection_mode}。仅支持 'all'、'distribution_shift'、'volume_spike'、'events_per_hour'。"

            out_dir = './agent_workspace'
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M')

            # 分别处理不同的数据需求
            mode_results = []
            target_columns = []

            # 处理需要原始事件数据的模式
            event_modes = [m for m in modes if m in ['distribution_shift', 'volume_spike']]
            if event_modes:
                try:
                    df, target_columns = fetch_fleet_data(fleet_id, start_date, end_date)

                    if not df.empty and target_columns:
                        for mode in event_modes:
                            res_df, csv_path, plot_path = self._run_one_mode(
                                mode, df, target_columns, fleet_id, timestamp, out_dir,
                                threshold_multiplier, min_abs_deviation, min_deviation,
                                trend_window, history_window, enable_visualization, spike_period,
                                stable_regime_points, stable_regime_tolerance_ratio, stable_shift_min_ratio,
                                expected_relative_tolerance, threshold_width_cap_ratio,
                                lower_anomaly_tolerance_multiplier, normal_dispersion_multiplier,
                                normal_dispersion_floor_ratio, normal_dispersion_min_points
                            )
                            mode_results.append((mode, res_df, csv_path, plot_path))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️ 获取事件数据失败: {e}")

            # 处理 events_per_hour 模式
            if 'events_per_hour' in modes:
                try:
                    df_eph = fetch_events_per_hour_data(start_date, end_date)
                    if not df_eph.empty:
                        res_df, csv_path, plot_path = self._run_events_per_hour_mode(
                            df_eph, "GroundCloud", timestamp, out_dir,
                            threshold_multiplier, min_abs_deviation, min_deviation,
                            trend_window, history_window, enable_visualization, spike_period,
                            stable_regime_points, stable_regime_tolerance_ratio, stable_shift_min_ratio,
                            expected_relative_tolerance, threshold_width_cap_ratio,
                            lower_anomaly_tolerance_multiplier, normal_dispersion_multiplier,
                            normal_dispersion_floor_ratio, normal_dispersion_min_points
                        )
                        mode_results.append(('events_per_hour', res_df, csv_path, plot_path))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️ 获取 events_per_hour 数据失败: {e}")

            if not mode_results:
                return f"⚠️ 无法检测：指定时间段内没有获取到有效数据。"

            last_event_date = mode_results[0][1]['event_date'].iloc[-1]
            last_day_anomalies = []
            for mode, res_df, _, _ in mode_results:
                last_row = res_df.iloc[-1]
                if bool(last_row['is_anomaly']):
                    last_day_anomalies.append((mode, last_row))

            anomaly_records = self._collect_last_day_anomaly_records(
                mode_results, target_columns, fleet_id, start_date, end_date, last_event_date
            )
            summary_path = f"{out_dir}/last_day_anomaly_summary_{fleet_id}_{timestamp}.json"
            structured_result = self._build_structured_result(
                fleet_id, start_date, end_date, last_event_date, modes,
                len(mode_results[0][1]), anomaly_records, summary_path
            )
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(structured_result, f, ensure_ascii=False, indent=2)

            report = [
                f"✅ 【事件窗口末日异常检测完成】",
                f"   - 扫描区间: {start_date} 至 {end_date} (共 {len(mode_results[0][1])} 天)",
                f"   - 判定日期: {last_event_date.strftime('%Y-%m-%d')}",
                f"   - 检测模式: {', '.join(modes)}",
                f"   - 可视化: {'已开启' if enable_visualization else '已关闭'}",
                f"   - 突增检测强制周期: {spike_period} 天",
                f"   - 稳定台阶切换: 最近连续 {stable_regime_points} 天稳定则允许基线切换",
                f"   - 历史回看趋势窗口: {trend_window} 天 | MAD 判定窗口: {history_window} 天",
                f"📊 事件窗口最后一天共被 {len(last_day_anomalies)} 个模式判定为异常，汇总到 {len(anomaly_records)} 条结构化异常信息。"
            ]

            if last_day_anomalies:
                report.append("🚨 末日异常详情:")
                for mode, row in sorted(last_day_anomalies, key=lambda item: item[1]['anomaly_score'], reverse=True):
                    report.append(f"  - 模式: {mode} | 严重度评分: {row['anomaly_score']:.2f}")
                    report.append(f"    诊断原因: {row['anomaly_reason']}")
            else:
                report.append("✅ 事件窗口最后一天未被任何检测模式判定为异常。")

            report.append("\n📂 详细评估底表:")
            if enable_visualization:
                for mode, _, csv_path, _ in mode_results:
                    report.append(f"  - {mode}: `{csv_path}`")
            else:
                report.append("  - 已关闭：enable_visualization=False 时不保存 CSV。")
            report.append(f"📌 末日异常汇总文件: `{summary_path}`")

            if enable_visualization:
                report.append("🎨 可视化图表:")
                for mode, _, _, plot_path in mode_results:
                    report.append(f"  - {mode}: `{plot_path}`")

            report.append("\nSTRUCTURED_ANOMALY_RESULT_JSON:")
            report.append(json.dumps(structured_result, ensure_ascii=False, indent=2))

            return "\n".join(report)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 工具执行期间发生严重错误: {str(e)}\n请检查参数或查询逻辑。"
