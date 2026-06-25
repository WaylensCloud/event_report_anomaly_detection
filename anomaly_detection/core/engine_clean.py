"""
Core anomaly detection engine.
核心异常检测引擎

包含所有异常检测算法和可视化方法。
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Type, List, Tuple
import warnings

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

# 尝试导入 scipy
try:
    from scipy.spatial.distance import jensenshannon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found, distribution shift detection will not work")

warnings.filterwarnings("ignore")


class AnomalyDetectorEngine:
    """
    核心异常检测引擎类
    提供多种异常检测算法和可视化方法
    """

    @staticmethod
    def auto_detect_period(series: pd.Series, max_lag: int = 40) -> Tuple[int, float]:
        """
        对历史序列寻找自相关周期

        Args:
            series: 时间序列
            max_lag: 最大滞后阶数

        Returns:
            (周期, 最大相关系数)
        """
        max_lag = min(max_lag, len(series) // 2)
        if len(series) < 14 or max_lag < 2:
            return 0, 0.0

        trend = series.rolling(window=21, min_periods=1, center=True).median()
        detrended = series - trend

        lags = range(2, max_lag + 1)
        corrs = [detrended.autocorr(lag=lag) for lag in lags]
        corrs = [c if pd.notna(c) else 0.0 for c in corrs]

        if not corrs:
            return 0, 0.0

        max_corr = max(corrs)
        if max_corr > 0.25:
            best_lag = lags[corrs.index(max_corr)]
            return best_lag, max_corr
        return 0, 0.0

    @staticmethod
    def _clean_phase_history_bidirectional(
        phase_history: pd.Series, threshold_multiplier: float, min_deviation: float
    ) -> pd.Series:
        """
        对同相位历史做双向鲁棒清洗，去掉历史异常点对预测的干扰

        Args:
            phase_history: 同相位历史数据
            threshold_multiplier: 阈值乘数
            min_deviation: 最小偏差

        Returns:
            清洗后的序列
        """
        y = pd.Series(phase_history).astype(float).reset_index(drop=True)
        if len(y) < 3:
            return y.clip(lower=0)

        clean = y.copy()
        for _ in range(2):
            if len(clean) >= 5:
                rolling = clean.rolling(5, center=True, min_periods=1).median()
            else:
                rolling = pd.Series(float(clean.median()), index=clean.index)

            residual = y - rolling
            center = float(residual.median())
            mad = float(np.median(np.abs(residual - center)))
            scale = max(mad * 1.4826, 1.0)
            tolerance = np.maximum.reduce([
                np.full(len(y), min_deviation, dtype=float),
                np.abs(rolling.to_numpy(dtype=float)) * 0.35,
                np.full(len(y), scale * threshold_multiplier, dtype=float),
            ])
            outlier_mask = np.abs(residual.to_numpy(dtype=float) - center) > tolerance
            clean.loc[outlier_mask] = rolling.loc[outlier_mask].clip(lower=0)

        return clean.clip(lower=0)

    @staticmethod
    def _predict_phase_next_value(clean_phase_history: pd.Series) -> Tuple[float, float]:
        """
        基于已确认正常历史趋势预测下一次出现的值

        Args:
            clean_phase_history: 清洗后的同相位历史数据

        Returns:
            (预测值, 预测尺度)
        """
        y = pd.Series(clean_phase_history).astype(float).reset_index(drop=True)
        if y.empty:
            return 0.0, 1.0
        if len(y) == 1:
            return max(float(y.iloc[-1]), 0.0), 1.0

        recent_window = min(8, len(y))
        recent = y.tail(recent_window)
        median_pred = float(recent.median())

        if len(recent) >= 3:
            x = np.arange(len(recent), dtype=float)
            try:
                slope, intercept = np.polyfit(x, recent.values, 1)
                trend_pred = float(intercept + slope * len(recent))
            except Exception:
                trend_pred = float(recent.median())
        else:
            trend_pred = float(recent.iloc[-1])

        last_value = float(recent.iloc[-1])
        median_abs_step = float(np.median(np.abs(np.diff(recent.values)))) if len(recent) >= 2 else 0.0
        step_guard = max(abs(last_value) * 0.25, median_abs_step * 2.0, 1.0)
        pred = float(np.clip(trend_pred, last_value - step_guard, last_value + step_guard))

        x_recent = np.arange(len(recent), dtype=float)
        try:
            slope_recent, intercept_recent = np.polyfit(x_recent, recent.values, 1)
            fitted = pd.Series(intercept_recent + slope_recent * x_recent, index=recent.index)
        except Exception:
            fitted = pd.Series(median_pred, index=recent.index)

        residual = recent - fitted
        center = float(residual.median())
        mad = float(np.median(np.abs(residual - center)))
        scale = max(mad * 1.4826, float(residual.std() or 0.0), 1.0)
        return max(pred, 0.0), scale

    @staticmethod
    def _candidate_segment_is_coherent(
        candidate_values: List[float],
        min_points: int,
        tolerance_ratio: float,
        min_deviation: float
    ) -> bool:
        """
        候选新趋势段内部足够自洽时，才允许它成为新的预测趋势

        Args:
            candidate_values: 候选值列表
            min_points: 最小点数
            tolerance_ratio: 容差比例
            min_deviation: 最小偏差

        Returns:
            是否自洽
        """
        min_points = max(int(min_points), 2)
        if len(candidate_values) < min_points:
            return False

        y = pd.Series(candidate_values[-min_points:]).astype(float).reset_index(drop=True)
        level = max(abs(float(y.median())), 1.0)
        absolute_floor = max(float(min_deviation) * 0.25, level * 0.02, 1e-9)
        if len(y) == 2:
            return abs(float(y.iloc[-1] - y.iloc[0])) <= max(
                level * tolerance_ratio,
                min_deviation * 0.25,
                absolute_floor
            )

        x = np.arange(len(y), dtype=float)
        try:
            slope, intercept = np.polyfit(x, y.values, 1)
            fitted = pd.Series(intercept + slope * x)
        except Exception:
            fitted = pd.Series(float(y.median()), index=y.index)

        residual = y - fitted
        max_residual = float(np.max(np.abs(residual)))
        step_median = float(np.median(np.abs(np.diff(y.values)))) if len(y) >= 2 else 0.0
        tolerance = max(level * tolerance_ratio, step_median * 1.5, min_deviation * 0.25, absolute_floor)
        return max_residual <= tolerance

    @staticmethod
    def _normal_dispersion_width(
        normal_history: pd.Series,
        expected: float,
        min_deviation: float,
        normal_dispersion_multiplier: float,
        normal_dispersion_floor_ratio: float,
        normal_dispersion_min_points: int
    ) -> float:
        """
        根据已确认的非异常历史数据估计正常离散范围。
        偏差如果仍落在这个范围内，就更像正常波动，而不是异常。
        """
        y = pd.Series(normal_history).astype(float).reset_index(drop=True)
        if len(y) < max(int(normal_dispersion_min_points), 2):
            return max(min_deviation, abs(expected) * normal_dispersion_floor_ratio, 1e-9)

        recent = y.tail(min(10, len(y)))
        if len(recent) >= 3:
            x = np.arange(len(recent), dtype=float)
            try:
                slope, intercept = np.polyfit(x, recent.values, 1)
                fitted = pd.Series(intercept + slope * x, index=recent.index)
            except Exception:
                fitted = pd.Series(float(recent.median()), index=recent.index)
        else:
            fitted = pd.Series(float(recent.median()), index=recent.index)

        residual = recent - fitted
        center = float(residual.median())
        mad_scale = float(np.median(np.abs(residual - center))) * 1.4826
        std_scale = float(residual.std() or 0.0)
        q75, q25 = np.percentile(residual, [75, 25])
        iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
        dispersion = max(mad_scale, std_scale, iqr_scale, 1e-9)

        return max(
            dispersion * normal_dispersion_multiplier,
            min_deviation,
            abs(expected) * normal_dispersion_floor_ratio,
            1e-9
        )

    @staticmethod
    def _run_periodic_phase_anomaly(
        series: pd.Series, threshold_multiplier: float, min_deviation: float,
        spike_period: int = 7, min_phase_history: int = 3,
        stable_regime_points: int = 3,
        stable_regime_tolerance_ratio: float = 0.10,
        stable_shift_min_ratio: float = 0.45,
        expected_relative_tolerance: float = 0.65,
        threshold_width_cap_ratio: float = 1.80,
        lower_anomaly_tolerance_multiplier: float = 1.80,
        normal_dispersion_multiplier: float = 3.0,
        normal_dispersion_floor_ratio: float = 0.25,
        normal_dispersion_min_points: int = 4
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        干净版强制周期相位异常检测。

        设计原则：
        1. 预测当前点时，只使用当前点之前、同相位、已确认正常的历史点。
        2. 当前点是否异常，只基于“旧趋势预测 + 阈值区间”判断。
        3. 异常点不会进入 confirmed 历史，避免污染下一次预测。
        4. 异常点只进入 candidate 候选新趋势；候选段稳定后，才切换后续预测趋势。
        5. 趋势切换只影响未来点，绝不回头取消当前点的异常标记。
        6. 当最近 confirmed 历史非常稳定时，额外用局部稳定带捕捉明显尖峰/断崖；
           但必须达到“趋势级变化幅度”，避免把稳定线附近小波动误报为异常。

        Args:
            series: 待检测序列
            threshold_multiplier: 阈值乘数
            min_deviation: 最小偏差
            spike_period: 周期长度
            min_phase_history: 最小相位历史长度
            stable_regime_points: 稳定区间点数
            stable_regime_tolerance_ratio: 稳定区间容差比例
            stable_shift_min_ratio: 稳定切换最小比例
            expected_relative_tolerance: 预测值附近的最小相对容忍区间
            threshold_width_cap_ratio: 阈值区间最大可扩到 expected 的比例
            lower_anomaly_tolerance_multiplier: 低于预测值时额外放宽倍数
            normal_dispersion_multiplier: 历史正常离散度放大倍数
            normal_dispersion_floor_ratio: 历史离散度阈值相对下限
            normal_dispersion_min_points: 启用历史离散度判断的最少已确认点数

        Returns:
            (异常标记, 异常分数, 预期值, 动态尺度, 上界)
        """
        original_index = series.index
        raw_y = pd.Series(series).astype(float).reset_index(drop=True)
        spike_period = max(int(spike_period), 1)
        min_phase_history = max(int(min_phase_history), 2)
        stable_regime_points = max(int(stable_regime_points), 2)

        expected_values = []
        threshold_width_values = []
        lower_threshold_width_values = []
        phase_history_counts = []
        cleaned_phase_tail_values = []
        regime_shift_flags = []
        regime_levels = []
        point_anomaly_flags = []
        anomaly_scores = []

        phase_states = {
            phase: {
                # confirmed 只保存已确认正常趋势点。预测永远只看它。
                "confirmed": [],
                # candidate 保存偏离旧趋势但可能形成新趋势的点。预测暂时不看它。
                "candidate": [],
                "candidate_direction": None,
                "using_new_trend": False,
            }
            for phase in range(spike_period)
        }

        for i in range(len(raw_y)):
            phase = i % spike_period
            actual = float(raw_y.iloc[i])
            state = phase_states[phase]
            confirmed_history = pd.Series(state["confirmed"], dtype=float)
            phase_history_counts.append(len(confirmed_history))

            if len(confirmed_history) == 0:
                # 没有历史时无法预测。为了画图连续，expected 放在当前真实值。
                expected = float(raw_y.iloc[i])
                scale = max(min_deviation, abs(expected) * normal_dispersion_floor_ratio, 1e-9)
                cleaned_phase_tail_values.append(np.nan)
            else:
                # 历史清洗只作用于已确认正常点，避免旧异常污染预测曲线。
                clean_phase_history = AnomalyDetectorEngine._clean_phase_history_bidirectional(
                    confirmed_history, threshold_multiplier, min_deviation
                )
                expected, scale = AnomalyDetectorEngine._predict_phase_next_value(clean_phase_history)
                cleaned_phase_tail_values.append(float(clean_phase_history.iloc[-1]))

            uses_new_trend = bool(state["using_new_trend"])
            regime_shift_flags.append(uses_new_trend)
            regime_levels.append(float(np.median(state["confirmed"])) if uses_new_trend and state["confirmed"] else np.nan)

            eligible = len(confirmed_history) >= min_phase_history

            # safe range 是普通异常判断区间。它允许中小波动，但不允许明显偏离旧趋势。
            dynamic_min = max(min_deviation, abs(expected) * expected_relative_tolerance)
            raw_width = max(scale * threshold_multiplier, 1e-9)
            width_cap = max(min_deviation, abs(expected) * threshold_width_cap_ratio)
            threshold_width = max(dynamic_min, min(raw_width, width_cap))
            lower_threshold_width = max(
                threshold_width * lower_anomaly_tolerance_multiplier,
                min_deviation * 1.5,
                abs(expected) * expected_relative_tolerance * lower_anomaly_tolerance_multiplier
            )

            expected_values.append(max(expected, 0.0))
            threshold_width_values.append(max(threshold_width, 1e-9))
            lower_threshold_width_values.append(max(lower_threshold_width, 1e-9))

            deviation = actual - expected
            abs_deviation = abs(deviation)
            score_scale_value = max(
                (threshold_width if deviation >= 0 else lower_threshold_width) / max(threshold_multiplier, 1e-6),
                1.0
            )

            # 1) 普通异常判断：当前点相对旧趋势预测是否落出 safe range。
            is_point_anomaly = False
            if eligible:
                if deviation >= 0:
                    is_point_anomaly = deviation > threshold_width
                else:
                    is_point_anomaly = abs_deviation > lower_threshold_width

            # 2) 局部稳定带强判定：
            #    只在最近 confirmed 本身很稳定时启用，用来拦截明显尖峰/断崖。
            #    同时要求变化达到 stable_shift_min_ratio 或足够大的绝对变化，避免小波动误报。
            if not is_point_anomaly and eligible and len(confirmed_history) >= min_phase_history:
                recent_confirmed = confirmed_history.tail(min(4, len(confirmed_history)))
                local_level = float(recent_confirmed.median())
                local_step = (
                    float(np.median(np.abs(np.diff(recent_confirmed.values))))
                    if len(recent_confirmed) >= 2 else 0.0
                )
                local_spread = float(np.max(recent_confirmed) - np.min(recent_confirmed))
                local_stability_width = max(
                    min_deviation,
                    abs(local_level) * stable_regime_tolerance_ratio * 1.5,
                    local_step * 3.0,
                    1e-9
                )
                is_recent_history_stable = local_spread <= max(local_stability_width * 1.5, min_deviation)
                if is_recent_history_stable:
                    upper_reference = max(expected, local_level)
                    lower_reference = min(expected, local_level)
                    material_abs_shift = max(min_deviation * 2.0, abs(local_level) * stable_shift_min_ratio)
                    if actual > upper_reference + local_stability_width:
                        if actual - upper_reference >= material_abs_shift:
                            is_point_anomaly = True
                            score_scale_value = max(local_stability_width / max(threshold_multiplier, 1e-6), 1.0)
                    elif actual < lower_reference - local_stability_width:
                        if lower_reference - actual >= material_abs_shift:
                            is_point_anomaly = True
                            score_scale_value = max(local_stability_width / max(threshold_multiplier, 1e-6), 1.0)

            point_anomaly_flags.append(is_point_anomaly)
            anomaly_scores.append(abs_deviation / score_scale_value if (eligible or is_point_anomaly) else 0.0)

            if is_point_anomaly:
                # 异常点不进入 confirmed，只进入候选新趋势。
                # 如果候选段连续稳定，后续预测切换到新趋势；当前点仍保持其旧趋势异常判断。
                anomaly_direction = "high" if deviation > 0 else "low"
                if state["candidate_direction"] != anomaly_direction:
                    state["candidate"] = []
                    state["candidate_direction"] = anomaly_direction
                state["candidate"].append(actual)
                max_candidate_len = max(stable_regime_points * 3, stable_regime_points)
                state["candidate"] = state["candidate"][-max_candidate_len:]
                if AnomalyDetectorEngine._candidate_segment_is_coherent(
                    state["candidate"],
                    stable_regime_points,
                    stable_regime_tolerance_ratio,
                    min_deviation
                ):
                    old_level = float(np.median(state["confirmed"])) if state["confirmed"] else 0.0
                    new_segment = state["candidate"][-stable_regime_points:]
                    new_level = float(np.median(new_segment))
                    shift_size = abs(new_level - old_level)
                    shift_floor = max(
                        min_deviation * 0.5,
                        max(abs(old_level), abs(new_level), 1e-9) * 0.05,
                        1e-9
                    )
                    shift_threshold = max(
                        shift_floor,
                        abs(old_level) * stable_shift_min_ratio,
                        abs(new_level) * stable_shift_min_ratio * 0.5
                    )
                    if shift_size >= shift_threshold:
                        state["confirmed"] = new_segment.copy()
                        state["candidate"] = []
                        state["candidate_direction"] = None
                        state["using_new_trend"] = True
            else:
                # 正常点才进入 confirmed，并且说明候选新趋势没有持续成立。
                state["confirmed"].append(actual)
                state["candidate"] = []
                state["candidate_direction"] = None

        expected_curve = pd.Series(expected_values, index=original_index).clip(lower=0)
        threshold_width = pd.Series(threshold_width_values, index=original_index).clip(lower=1e-9)
        lower_threshold_width = pd.Series(lower_threshold_width_values, index=original_index).clip(lower=1e-9)
        dynamic_scale = (threshold_width / max(threshold_multiplier, 1e-6)).clip(lower=1e-9)
        is_anomaly = pd.Series(point_anomaly_flags, index=original_index)
        anomaly_score = pd.Series(anomaly_scores, index=original_index)

        upper_bound = expected_curve + threshold_width
        lower_bound = (expected_curve - lower_threshold_width).clip(lower=0)

        expected_curve.attrs['model_name'] = 'periodic_phase_clean_state_machine'
        expected_curve.attrs['trained_inlier_counts'] = phase_history_counts
        expected_curve.attrs['lower_bound'] = lower_bound
        expected_curve.attrs['spike_period'] = spike_period
        expected_curve.attrs['cleaned_series'] = pd.Series(cleaned_phase_tail_values, index=original_index)
        expected_curve.attrs['regime_shift_flags'] = regime_shift_flags
        expected_curve.attrs['regime_levels'] = regime_levels

        return is_anomaly, anomaly_score, expected_curve, dynamic_scale, upper_bound

    # ------------------- 模式 1: 单项及总量因果突增检测 -------------------
    @staticmethod
    def run_volume_spike_sliding(
        df: pd.DataFrame, target_cols: List[str],
        threshold_multiplier: float, min_deviation: float,
        trend_window: int, history_window: int, spike_period: int = 7,
        stable_regime_points: int = 3,
        stable_regime_tolerance_ratio: float = 0.10,
        stable_shift_min_ratio: float = 0.45,
        expected_relative_tolerance: float = 0.65,
        threshold_width_cap_ratio: float = 1.80,
        lower_anomaly_tolerance_multiplier: float = 1.80,
        normal_dispersion_multiplier: float = 3.0,
        normal_dispersion_floor_ratio: float = 0.25,
        normal_dispersion_min_points: int = 4
    ) -> pd.DataFrame:
        """
        运行总量和单项突增异常检测

        Args:
            df: 数据框
            target_cols: 目标列
            threshold_multiplier: 阈值乘数
            min_deviation: 最小偏差
            trend_window: 趋势窗口
            history_window: 历史窗口
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
            检测结果数据框
        """
        df = df.copy()
        df['total'] = df[target_cols].sum(axis=1)
        df['spike_period'] = max(int(spike_period), 1)

        # 1. 扫描大盘总量
        tot_anom, tot_score, exp_tot, sc_tot, ub_tot = AnomalyDetectorEngine._run_periodic_phase_anomaly(
            df['total'], threshold_multiplier, min_deviation, spike_period,
            stable_regime_points=stable_regime_points,
            stable_regime_tolerance_ratio=stable_regime_tolerance_ratio,
            stable_shift_min_ratio=stable_shift_min_ratio,
            expected_relative_tolerance=expected_relative_tolerance,
            threshold_width_cap_ratio=threshold_width_cap_ratio,
            lower_anomaly_tolerance_multiplier=lower_anomaly_tolerance_multiplier,
            normal_dispersion_multiplier=normal_dispersion_multiplier,
            normal_dispersion_floor_ratio=normal_dispersion_floor_ratio,
            normal_dispersion_min_points=normal_dispersion_min_points
        )
        df['expected_total'] = exp_tot
        df['total_lower_bound'] = exp_tot.attrs.get('lower_bound', pd.Series(0.0, index=df.index))
        df['total_upper_bound'] = ub_tot
        df['is_total_anomaly'] = tot_anom
        df['total_ml_train_inliers'] = exp_tot.attrs.get('trained_inlier_counts', [0] * len(df))
        df['total_cleaned_for_model'] = exp_tot.attrs.get('cleaned_series', df['total'])
        df['total_regime_shift_active'] = exp_tot.attrs.get('regime_shift_flags', [False] * len(df))
        df['total_regime_level'] = exp_tot.attrs.get('regime_levels', [np.nan] * len(df))

        # 2. 扫描所有独立单项
        any_sub_anom = np.zeros(len(df), dtype=bool)
        anom_reasons = [''] * len(df)
        max_scores = tot_score.copy()

        sub_results = {}
        for col in target_cols:
            if df[col].max() < 1:
                continue
            sub_anom, sub_score, sub_exp, _, sub_ub = AnomalyDetectorEngine._run_periodic_phase_anomaly(
                df[col], threshold_multiplier, min_deviation, spike_period,
                stable_regime_points=stable_regime_points,
                stable_regime_tolerance_ratio=stable_regime_tolerance_ratio,
                stable_shift_min_ratio=stable_shift_min_ratio,
                expected_relative_tolerance=expected_relative_tolerance,
                threshold_width_cap_ratio=threshold_width_cap_ratio,
                lower_anomaly_tolerance_multiplier=lower_anomaly_tolerance_multiplier,
                normal_dispersion_multiplier=normal_dispersion_multiplier,
                normal_dispersion_floor_ratio=normal_dispersion_floor_ratio,
                normal_dispersion_min_points=normal_dispersion_min_points
            )
            sub_results[col] = (sub_anom, sub_score, sub_exp)

            # 如果该单项在整个扫描周期内发生过异常，将其完整信息保存到 df 中供独立作图
            if sub_anom.any():
                df[f'sub_exp_{col}'] = sub_exp
                df[f'sub_lb_{col}'] = sub_exp.attrs.get('lower_bound', pd.Series(0.0, index=df.index))
                df[f'sub_ub_{col}'] = sub_ub
                df[f'sub_anom_{col}'] = sub_anom
                df[f'sub_score_{col}'] = sub_score
                df[f'sub_ml_train_inliers_{col}'] = sub_exp.attrs.get('trained_inlier_counts', [0] * len(df))
                df[f'sub_cleaned_for_model_{col}'] = sub_exp.attrs.get('cleaned_series', df[col])
                df[f'sub_regime_shift_active_{col}'] = sub_exp.attrs.get('regime_shift_flags', [False] * len(df))
                df[f'sub_regime_level_{col}'] = sub_exp.attrs.get('regime_levels', [np.nan] * len(df))

        # 3. 汇总报警理由
        for i in range(len(df)):
            reasons = []
            if tot_anom.iloc[i]:
                reasons.append(f"【总量骤增】实际 {df['total'].iloc[i]:.0f}次 (预期仅 {exp_tot.iloc[i]:.0f}次)")

            for col, (sub_anom, sub_score, sub_exp) in sub_results.items():
                if sub_anom.iloc[i]:
                    reasons.append(f"【{col}突增】实际 {df[col].iloc[i]:.0f}次 (预期仅 {sub_exp.iloc[i]:.0f}次)")
                    any_sub_anom[i] = True
                    if sub_score.iloc[i] > max_scores.iloc[i]:
                        max_scores.iloc[i] = sub_score.iloc[i]

            if reasons:
                anom_reasons[i] = "；".join(reasons)

        df['is_anomaly'] = tot_anom | any_sub_anom
        df['is_sub_anomaly_only'] = any_sub_anom & (~tot_anom)
        df['anomaly_reason'] = anom_reasons
        df['anomaly_score'] = max_scores
        return df

    @staticmethod
    def plot_volume_spike_sliding(df: pd.DataFrame, fleet_id: str, save_path: str, spike_period: int = 7):
        """
        生成总量与单项突增智能可视化面板。
        如果有单项指标异常，会自动拆分出专属的子图，防止被大盘量级掩蔽。
        """
        dates = df['event_date']
        if 'spike_period' in df.columns and len(df) > 0:
            spike_period = int(df['spike_period'].iloc[0])
        spike_period = max(int(spike_period), 1)

        def plot_period_split(fig, subplot_spec, value_col, exp_col, anom_col, title, ylabel):
            sub_gs = GridSpecFromSubplotSpec(
                spike_period, 1, subplot_spec=subplot_spec, hspace=0.16
            )
            cmap = plt.get_cmap('tab10' if spike_period <= 10 else 'tab20')
            first_ax = None
            last_ax = None
            axes = []
            for phase in range(spike_period):
                phase_positions = [i for i in range(len(df)) if i % spike_period == phase]
                ax = fig.add_subplot(sub_gs[phase, 0], sharex=first_ax)
                axes.append(ax)
                if first_ax is None:
                    first_ax = ax
                last_ax = ax

                if not phase_positions:
                    ax.text(0.01, 0.55, f'Phase {phase + 1}: no data', transform=ax.transAxes, fontsize=9)
                    ax.grid(True, alpha=0.25)
                    continue

                phase_df = df.iloc[phase_positions]
                color = cmap(phase % cmap.N)
                ax.plot(
                    phase_df['event_date'], phase_df[value_col], marker='.', linewidth=1.2,
                    alpha=0.80, color=color, label=f'Phase {phase + 1} Actual'
                )
                ax.plot(
                    phase_df['event_date'], phase_df[exp_col], linestyle='--', linewidth=1.2,
                    alpha=0.95, color=color, label=f'Phase {phase + 1} Expected'
                )

                if anom_col in phase_df.columns:
                    anomalies = phase_df[phase_df[anom_col]]
                    if not anomalies.empty:
                        ax.scatter(
                            anomalies['event_date'], anomalies[value_col],
                            color='red', s=46, marker='^', zorder=10
                        )

                ax.text(
                    0.01, 0.78, f'Phase {phase + 1}',
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    color=color, bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1.5)
                )
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, alpha=0.25)
                if phase == 0:
                    ax.legend(loc='upper right', fontsize=8, ncol=2)

            if first_ax is not None:
                first_ax.set_title(title, fontsize=13, fontweight='bold', color='darkslategray')
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
            if last_ax is not None:
                last_ax.set_xlabel('Event date', fontsize=10)

        # 提取所有发生过异常的单项事件
        anomalous_items = [c.replace('sub_anom_', '') for c in df.columns if c.startswith('sub_anom_')]

        # 为了防止图表过长，如果异常单项过多，按异常严重度截取前 4 个最显著的
        if anomalous_items:
            item_max_scores = {col: df[f'sub_score_{col}'].max() for col in anomalous_items}
            sorted_items = sorted(item_max_scores.keys(), key=lambda k: item_max_scores[k], reverse=True)
            top_items = sorted_items[:4]
        else:
            top_items = []

        phase_section_height = max(3.6, spike_period * 0.9)
        section_heights = [1.4, phase_section_height]
        for _ in top_items:
            section_heights.extend([1.4, phase_section_height])

        fig_height = max(10, 4.0 * sum(section_heights))
        fig = plt.figure(figsize=(16, fig_height), constrained_layout=True)
        gs = plt.GridSpec(len(section_heights), 1, height_ratios=section_heights)

        # --- Plot 1: 永远保留的大盘总量趋势图 ---
        ax0 = fig.add_subplot(gs[0])
        safe_lower = (df['expected_total'] - (df['total_upper_bound'] - df['expected_total'])).clip(lower=0)
        ax0.fill_between(dates, safe_lower, df['total_upper_bound'], color='green', alpha=0.15, label='Safe Range (Expected Total)')
        ax0.plot(dates, df['expected_total'], color='green', linestyle='--', linewidth=2, label='Expected Trend Baseline')
        ax0.plot(dates, df['total'], color='royalblue', linewidth=1.5, marker='.', alpha=0.8, label='Actual Total Events')

        anomalies_total = df[df['is_total_anomaly']]
        if not anomalies_total.empty:
            ax0.scatter(anomalies_total['event_date'], anomalies_total['total'],
                       color='red', s=120, marker='X', zorder=10, label='Total Volume Spike')
            top_anoms = anomalies_total.sort_values('anomaly_score', ascending=False).head(5)
            for _, row in top_anoms.iterrows():
                ax0.annotate(row['event_date'].strftime('%m-%d'), (row['event_date'], row['total']),
                            xytext=(5, 5), textcoords='offset points', color='darkred', fontsize=9, fontweight='bold')

        ax0.set_title(f'1. Overall Volume Trend', fontsize=15, fontweight='bold', color='black')
        ax0.set_ylabel('Total Count', fontsize=12)
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc='upper right', fontsize=11)

        plot_period_split(
            fig,
            gs[1],
            'total',
            'expected_total',
            'is_total_anomaly',
            f'2. Period-Split Overall Volume (period={spike_period})',
            'Total Count'
        )

        # --- Plot 2~N: 独立分离出的发生激增的单项图 ---
        for idx, col in enumerate(top_items):
            ax_sub = fig.add_subplot(gs[2 + idx * 2])
            exp_col, ub_col, anom_col = f'sub_exp_{col}', f'sub_ub_{col}', f'sub_anom_{col}'

            safe_lower_sub = (df[exp_col] - (df[ub_col] - df[exp_col])).clip(lower=0)
            ax_sub.fill_between(dates, safe_lower_sub, df[ub_col], color='teal', alpha=0.15, label=f'Safe Range ({col})')
            ax_sub.plot(dates, df[exp_col], color='teal', linestyle='--', linewidth=2, label='Expected Baseline')
            ax_sub.plot(dates, df[col], color='darkorange', linewidth=1.5, marker='.', alpha=0.8, label=f'Actual {col}')

            anomalies_sub = df[df[anom_col]]
            if not anomalies_sub.empty:
                ax_sub.scatter(anomalies_sub['event_date'], anomalies_sub[col],
                           color='red', s=100, marker='^', zorder=10, label=f'{col} Spike')
                # 标注具体的数字方便复盘
                top_sub = anomalies_sub.sort_values(f'sub_score_{col}', ascending=False).head(5)
                for _, row in top_sub.iterrows():
                    ax_sub.annotate(f"{row['event_date'].strftime('%m-%d')}\n({row[col]:.0f} vs {row[exp_col]:.0f})",
                                (row['event_date'], row[col]), xytext=(5, 5), textcoords='offset points',
                                color='darkred', fontsize=9, fontweight='bold')

            ax_sub.set_title(f'{3 + idx * 2}. Individual Item Spike: {col}', fontsize=14, fontweight='bold', color='darkred')
            ax_sub.set_ylabel('Event Count', fontsize=12)
            ax_sub.grid(True, alpha=0.3)
            ax_sub.legend(loc='upper right', fontsize=11)

            plot_period_split(
                fig,
                gs[3 + idx * 2],
                col,
                exp_col,
                anom_col,
                f'{4 + idx * 2}. Period-Split Item: {col} (period={spike_period})',
                'Event Count'
            )

        fig.suptitle(f'Continuous Event Analysis - Fleet: {fleet_id}', fontsize=18, fontweight='bold', y=1.0)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ------------------- 模式 2: 分布占比连续滑动检测 (JS 散度逻辑) -------------------
    @staticmethod
    def run_distribution_shift_sliding(
        df: pd.DataFrame, target_cols: List[str],
        threshold_multiplier: float, min_divergence: float, trend_window: int, history_window: int
    ) -> pd.DataFrame:
        """
        运行分布漂移异常检测

        Args:
            df: 数据框
            target_cols: 目标列
            threshold_multiplier: 阈值乘数
            min_divergence: 最小散度
            trend_window: 趋势窗口
            history_window: 历史窗口

        Returns:
            检测结果数据框
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy is required for distribution shift detection")

        df = df.copy()
        df['total'] = df[target_cols].sum(axis=1)
        props = df[target_cols].div(df['total'].replace(0, 1e-9), axis=0).fillna(0.0)

        for col in target_cols:
            df[col + '_prop'] = props[col]

        expected_props = pd.DataFrame(index=df.index, columns=target_cols)

        for col in target_cols:
            if props[col].max() < 0.01:
                expected_props[col] = props[col].shift(1).rolling(trend_window, min_periods=1).median().bfill()
                continue

            period, _ = AnomalyDetectorEngine.auto_detect_period(props[col])
            trend = props[col].shift(1).rolling(trend_window, min_periods=1).median().bfill()
            seasonality = np.zeros(len(props))

            if period > 0:
                detrended = props[col] - trend
                lags = [period * i for i in range(1, 5)]
                lag_df = pd.DataFrame({f'lag_{lag}': detrended.shift(lag) for lag in lags})
                seasonality = lag_df.median(axis=1).fillna(0)

            expected_props[col] = (trend + seasonality).clip(lower=0.0, upper=1.0)

        row_sums = expected_props.sum(axis=1).replace(0, 1e-9)
        expected_props = expected_props.div(row_sums, axis=0)
        for col in target_cols:
            df[f'{col}_expected_prop'] = expected_props[col]

        js_divs = []
        for i in range(len(df)):
            p_act = props.iloc[i].values + 1e-9
            p_exp = expected_props.iloc[i].values + 1e-9
            js = jensenshannon(p_act / p_act.sum(), p_exp / p_exp.sum(), base=2.0)
            js_divs.append(js)

        df['js_divergence'] = js_divs

        def calc_mad(x):
            return np.median(np.abs(x - np.median(x))) if len(x) > 0 else 0.0

        baseline = df['js_divergence'].shift(1).rolling(history_window, min_periods=5).median().bfill().fillna(0)
        rolling_mad = df['js_divergence'].shift(1).rolling(history_window, min_periods=5).apply(calc_mad, raw=True)
        scale = (rolling_mad * 1.4826).bfill().clip(lower=0.01)

        df['baseline_divergence'] = baseline
        df['dynamic_scale'] = scale
        df['upper_bound'] = np.maximum(baseline + scale * threshold_multiplier, min_divergence)

        df['is_anomaly'] = df['js_divergence'] > df['upper_bound']
        df['anomaly_score'] = (df['js_divergence'] - baseline) / scale

        reasons = []
        for i in range(len(df)):
            if df['is_anomaly'].iloc[i]:
                top3 = props.iloc[i].astype(float).nlargest(3)
                desc = ", ".join([f"{idx} ({val*100:.1f}%)" for idx, val in top3.items()])
                reasons.append(f"结构崩塌主因: {desc}")
            else:
                reasons.append("")
        df['anomaly_reason'] = reasons

        return df

    @staticmethod
    def plot_distribution_shift_sliding(df: pd.DataFrame, target_cols: List[str], fleet_id: str, save_path: str):
        """
        生成分布漂移回测可视化大图
        """
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = plt.GridSpec(2, 1, height_ratios=[1.2, 2.5])

        dates = df['event_date']
        is_anomaly = df['is_anomaly']

        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(dates, 0, df['upper_bound'], color='green', alpha=0.15, label='Normal Distribution Shift Range')
        ax1.plot(dates, df['js_divergence'], color='purple', linewidth=1.5, marker='.', label='Actual JS Divergence')
        ax1.plot(dates, df['upper_bound'], color='red', linestyle='--', linewidth=1.5, label='Dynamic Anomaly Threshold')

        if np.sum(is_anomaly) > 0:
            ax1.scatter(dates[is_anomaly], df['js_divergence'][is_anomaly], color='red', s=100, zorder=10, label='Distribution Breakdown!', marker='X')

        ax1.set_title(f'Continuous Distribution Shift (JS Divergence) - Fleet: {fleet_id}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('JS Divergence (0~1)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1])
        prop_cols = [c + '_prop' for c in target_cols]
        mean_props = df[prop_cols].mean().sort_values(ascending=False)
        top_events = mean_props.head(6).index.tolist()

        plot_df = df[top_events].copy()
        plot_df['Others'] = 1.0 - plot_df.sum(axis=1)
        plot_df['Others'] = plot_df['Others'].clip(lower=0)

        labels = [c.replace('_prop', '') for c in plot_df.columns]
        ax2.stackplot(dates, plot_df.T.values * 100, labels=labels, alpha=0.85)

        for idx, anomaly_flag in enumerate(is_anomaly):
            if anomaly_flag:
                ax2.axvline(dates.iloc[idx], color='red', linestyle=':', linewidth=2, zorder=10)

        ax2.set_title('Daily Event Composition Trend (100% Stacked Area)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Proportion (%)', fontsize=12)
        ax2.margins(x=0, y=0)
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ------------------- 模式 3: events_per_hour 比率异常检测 -------------------
    @staticmethod
    def run_events_per_hour_spike(
        df: pd.DataFrame,
        threshold_multiplier: float,
        min_deviation: float,
        trend_window: int,
        history_window: int,
        spike_period: int = 7,
        stable_regime_points: int = 3,
        stable_regime_tolerance_ratio: float = 0.10,
        stable_shift_min_ratio: float = 0.45,
        expected_relative_tolerance: float = 0.65,
        threshold_width_cap_ratio: float = 1.80,
        lower_anomaly_tolerance_multiplier: float = 1.80,
        normal_dispersion_multiplier: float = 3.0,
        normal_dispersion_floor_ratio: float = 0.25,
        normal_dispersion_min_points: int = 4
    ) -> pd.DataFrame:
        """
        专门检测 events_per_hour 指标的异常，使用与 volume_spike 相同的方法

        Args:
            df: 数据框（必须包含 events_per_hour 列）
            threshold_multiplier: 阈值乘数
            min_deviation: 最小偏差
            trend_window: 趋势窗口
            history_window: 历史窗口
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
            检测结果数据框
        """
        df = df.copy()

        # 直接对 events_per_hour 进行检测
        anom, score, exp, scale, ub = AnomalyDetectorEngine._run_periodic_phase_anomaly(
            df['events_per_hour'], threshold_multiplier, min_deviation, spike_period,
            stable_regime_points=stable_regime_points,
            stable_regime_tolerance_ratio=stable_regime_tolerance_ratio,
            stable_shift_min_ratio=stable_shift_min_ratio,
            expected_relative_tolerance=expected_relative_tolerance,
            threshold_width_cap_ratio=threshold_width_cap_ratio,
            lower_anomaly_tolerance_multiplier=lower_anomaly_tolerance_multiplier,
            normal_dispersion_multiplier=normal_dispersion_multiplier,
            normal_dispersion_floor_ratio=normal_dispersion_floor_ratio,
            normal_dispersion_min_points=normal_dispersion_min_points
        )

        df['expected_events_per_hour'] = exp
        df['events_per_hour_lower_bound'] = exp.attrs.get('lower_bound', pd.Series(0.0, index=df.index))
        df['events_per_hour_upper_bound'] = ub
        df['is_anomaly'] = anom
        df['anomaly_score'] = score
        df['events_per_hour_ml_train_inliers'] = exp.attrs.get('trained_inlier_counts', [0] * len(df))
        df['events_per_hour_cleaned_for_model'] = exp.attrs.get('cleaned_series', df['events_per_hour'])
        df['events_per_hour_regime_shift_active'] = exp.attrs.get('regime_shift_flags', [False] * len(df))
        df['events_per_hour_regime_level'] = exp.attrs.get('regime_levels', [np.nan] * len(df))
        df['spike_period'] = max(int(spike_period), 1)

        # 构建异常原因描述
        reasons = []
        for i in range(len(df)):
            if bool(anom.iloc[i]):
                actual = df['events_per_hour'].iloc[i]
                expected_val = exp.iloc[i]
                reasons.append(f"【每小时事件数异常】实际 {actual:.2f} 次/小时 (预期仅 {expected_val:.2f} 次/小时)")
            else:
                reasons.append("")
        df['anomaly_reason'] = reasons

        return df

    @staticmethod
    def plot_events_per_hour_spike(df: pd.DataFrame, fleet_id: str, save_path: str, spike_period: int = 7):
        """
        生成 events_per_hour 异常检测的可视化图表
        """
        dates = df['event_date']
        if 'spike_period' in df.columns and len(df) > 0:
            spike_period = int(df['spike_period'].iloc[0])
        spike_period = max(int(spike_period), 1)

        def plot_period_split(fig, subplot_spec, value_col, exp_col, anom_col, title, ylabel):
            sub_gs = GridSpecFromSubplotSpec(
                spike_period, 1, subplot_spec=subplot_spec, hspace=0.16
            )
            cmap = plt.get_cmap('tab10' if spike_period <= 10 else 'tab20')
            first_ax = None
            last_ax = None
            axes = []
            for phase in range(spike_period):
                phase_positions = [i for i in range(len(df)) if i % spike_period == phase]
                ax = fig.add_subplot(sub_gs[phase, 0], sharex=first_ax)
                axes.append(ax)
                if first_ax is None:
                    first_ax = ax
                last_ax = ax

                if not phase_positions:
                    ax.text(0.01, 0.78, f'Phase {phase + 1}: no data', transform=ax.transAxes, fontsize=9)
                    ax.grid(True, alpha=0.25)
                    continue

                phase_df = df.iloc[phase_positions]
                color = cmap(phase % cmap.N)
                ax.plot(
                    phase_df['event_date'], phase_df[value_col], marker='.', linewidth=1.2,
                    alpha=0.80, color=color, label=f'Phase {phase + 1} Actual'
                )
                ax.plot(
                    phase_df['event_date'], phase_df[exp_col], linestyle='--', linewidth=1.2,
                    alpha=0.95, color=color, label=f'Phase {phase + 1} Expected'
                )

                if anom_col in phase_df.columns:
                    anomalies = phase_df[phase_df[anom_col]]
                    if not anomalies.empty:
                        ax.scatter(
                            anomalies['event_date'], anomalies[value_col],
                            color='red', s=46, marker='^', zorder=10
                        )

                ax.text(
                    0.01, 0.78, f'Phase {phase + 1}',
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    color=color, bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1.5)
                )
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, alpha=0.25)
                if phase == 0:
                    ax.legend(loc='upper right', fontsize=8, ncol=2)

            if first_ax is not None:
                first_ax.set_title(title, fontsize=13, fontweight='bold', color='darkslategray')
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
            if last_ax is not None:
                last_ax.set_xlabel('Event Date', fontsize=10)

        phase_section_height = max(3.6, spike_period * 0.9)
        section_heights = [1.4, phase_section_height]

        fig_height = max(10, 4.0 * sum(section_heights))
        fig = plt.figure(figsize=(16, fig_height), constrained_layout=True)
        gs = plt.GridSpec(len(section_heights), 1, height_ratios=section_heights)

        # 主趋势图
        ax0 = fig.add_subplot(gs[0])
        safe_lower = (df['expected_events_per_hour'] - (df['events_per_hour_upper_bound'] - df['expected_events_per_hour'])).clip(lower=0)
        ax0.fill_between(dates, safe_lower, df['events_per_hour_upper_bound'], color='green', alpha=0.15, label='Safe Range (Expected)')
        ax0.plot(dates, df['expected_events_per_hour'], color='green', linestyle='--', linewidth=2, label='Expected Trend Baseline')
        ax0.plot(dates, df['events_per_hour'], color='royalblue', linewidth=1.5, marker='.', alpha=0.8, label='Actual Events/Hour')

        anomalies = df[df['is_anomaly']]
        if not anomalies.empty:
            ax0.scatter(anomalies['event_date'], anomalies['events_per_hour'],
                       color='red', s=120, marker='X', zorder=10, label='Events/Hour Anomaly')
            top_anoms = anomalies.sort_values('anomaly_score', ascending=False).head(5)
            for _, row in top_anoms.iterrows():
                ax0.annotate(row['event_date'].strftime('%m-%d'), (row['event_date'], row['events_per_hour']),
                            xytext=(5, 5), textcoords='offset points', color='darkred', fontsize=9, fontweight='bold')

        ax0.set_title(f'1. Events Per Hour Trend - Fleet: {fleet_id}', fontsize=15, fontweight='bold', color='black')
        ax0.set_ylabel('Events Per Hour', fontsize=12)
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc='upper right', fontsize=11)

        # 周期拆分图
        plot_period_split(
            fig,
            gs[1],
            'events_per_hour',
            'expected_events_per_hour',
            'is_anomaly',
            f'2. Period-Split Events Per Hour (period={spike_period})',
            'Events Per Hour'
        )

        fig.suptitle(f'Events Per Hour Anomaly Analysis - Fleet: {fleet_id}', fontsize=18, fontweight='bold', y=1.0)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
