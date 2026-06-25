#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列异常检测工具
【长周期滑动扫描与可视化双模式版】

核心特性：
1. 【distribution_shift】：保留原有的 JS 散度算法，检测每天的结构占比是否发生分布漂移崩塌。
2. 【volume_spike】：严格因果预测内核 (Causal Expected Curve)。
    - 对大盘总量 (Total) 和每个单项 (Item) 均执行趋势+周期分离的预测。
    - 采用动态 MAD 计算历史容忍度，杜绝未来数据穿越。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Type, List, Tuple
from datetime import datetime
import warnings

import matplotlib.pyplot as plt

# 导入 LangChain 工具定义的标准库
from pydantic import BaseModel, Field, root_validator
try:
    from langchain_core.tools import BaseTool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    class BaseTool: pass

try:
    from scipy.spatial.distance import jensenshannon
except ImportError:
    print("⚠️ scipy not found! Run: pip install scipy")

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import HuberRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not found! Run: pip install scikit-learn")

# 数据库连接依赖
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning if HAS_SKLEARN else Warning)

DEFAULT_TIME_WINDOW_DAYS = 90


def _today_str() -> str:
    return pd.Timestamp.today().strftime('%Y-%m-%d')


def _date_days_ago_str(time_window_days: int, end_date: str = None) -> str:
    end = pd.to_datetime(end_date or _today_str()).normalize()
    return (end - pd.Timedelta(days=int(time_window_days))).strftime('%Y-%m-%d')


def build_time_window_params(time_window_days: int = DEFAULT_TIME_WINDOW_DAYS) -> Dict[str, Any]:
    end_date = _today_str()
    return {
        "time_window_days": time_window_days,
        "start_date": _date_days_ago_str(time_window_days, end_date),
        "end_date": end_date
    }

# ==========================================
# 1. 核心算法引擎 (长周期滑动窗口全景扫描)
# ==========================================
class AnomalyDetectorEngine:

    @staticmethod
    def auto_detect_period(series: pd.Series, max_lag: int = 40) -> Tuple[int, float]:
        """对历史序列寻找自相关周期"""
        max_lag = min(max_lag, len(series) // 2)
        if len(series) < 14 or max_lag < 2:  
            return 0, 0.0
            
        trend = series.rolling(window=21, min_periods=1, center=True).median()
        detrended = series - trend
        
        lags = range(2, max_lag + 1)
        corrs = [detrended.autocorr(lag=lag) for lag in lags]
        corrs = [c if pd.notna(c) else 0.0 for c in corrs]
        
        if not corrs: return 0, 0.0
            
        max_corr = max(corrs)
        if max_corr > 0.25:
            best_lag = lags[corrs.index(max_corr)]
            return best_lag, max_corr
        return 0, 0.0

    @staticmethod
    def _build_causal_clean_series(
        series: pd.Series, trend_window: int, history_window: int,
        threshold_multiplier: float, min_deviation: float
    ) -> pd.Series:
        """
        生成因果清洗序列：发现历史点明显偏离之前的主体规律时，用之前历史的鲁棒基线替换。
        后续模型只学习这条 cleaned series，避免异常尖峰进入 lag/rolling 特征和训练目标。
        """
        y = pd.Series(series).astype(float).reset_index(drop=True)
        clean_values = []
        clean_residuals = []
        deviation_direction = 0
        deviation_streak = 0

        for i, actual in enumerate(y):
            if i == 0:
                clean_values.append(float(actual))
                clean_residuals.append(0.0)
                continue

            recent = pd.Series(clean_values[max(0, i - trend_window):i])
            trend_baseline = float(recent.median()) if not recent.empty else float(y.iloc[:i].median())

            seasonal_candidates = []
            for lag in (7, 14, 21):
                if i - lag >= 0:
                    seasonal_candidates.append(clean_values[i - lag])
            if seasonal_candidates:
                seasonal_baseline = float(np.median(seasonal_candidates))
                seasonal_mad = float(np.median(np.abs(np.array(seasonal_candidates) - seasonal_baseline)))
                seasonal_is_stable = seasonal_mad <= max(min_deviation, seasonal_baseline * 0.35, 1.0)
                if seasonal_is_stable:
                    baseline = seasonal_baseline
                else:
                    baseline = 0.45 * trend_baseline + 0.55 * seasonal_baseline
            else:
                baseline = trend_baseline

            residual_history = pd.Series(clean_residuals[max(0, i - history_window):i])
            if len(residual_history) >= 5:
                center = float(residual_history.median())
                mad = float(np.median(np.abs(residual_history - center)))
                robust_scale = max(mad * 1.4826, float(residual_history.std() or 0.0), 1.0)
            else:
                robust_scale = max(float(recent.std() or 0.0), 1.0)

            actual = float(actual)
            tolerance = max(min_deviation, baseline * 0.30, robust_scale * threshold_multiplier)
            deviation = actual - baseline
            is_deviation = abs(deviation) > tolerance
            recurring_weekly_pattern = False
            if seasonal_candidates:
                seasonal_tolerance = max(min_deviation, seasonal_baseline * 0.40, seasonal_mad * 3.0, 1.0)
                recurring_weekly_pattern = abs(actual - seasonal_baseline) <= seasonal_tolerance
                if recurring_weekly_pattern:
                    is_deviation = False

            if is_deviation:
                current_direction = 1 if deviation > 0 else -1
                if current_direction == deviation_direction:
                    deviation_streak += 1
                else:
                    deviation_direction = current_direction
                    deviation_streak = 1
            else:
                deviation_direction = 0
                deviation_streak = 0

            # 短期尖峰/塌陷继续按异常处理；持续偏移达到 5 天后，才承认为新规律。
            if is_deviation and deviation_streak < 5:
                clean_value = baseline
            else:
                clean_value = actual

            clean_values.append(max(clean_value, 0.0))
            clean_residuals.append(clean_values[-1] - baseline)

        return pd.Series(clean_values, index=series.index)

    @staticmethod
    def _causal_weekly_baseline(clean_series: pd.Series, trend_window: int) -> pd.Series:
        """用 cleaned history 中的同星期历史生成鲁棒周周期基线。"""
        y = pd.Series(clean_series).astype(float).reset_index(drop=True)
        values = []
        for i in range(len(y)):
            seasonal = [y.iloc[i - lag] for lag in (7, 14, 21, 28) if i - lag >= 0]
            recent = y.iloc[max(0, i - trend_window):i]

            if seasonal:
                seasonal_value = float(np.median(seasonal))
                seasonal_mad = float(np.median(np.abs(np.array(seasonal) - seasonal_value)))
                if not recent.empty:
                    recent_value = float(recent.median())
                    if seasonal_mad <= max(seasonal_value * 0.35, 1.0):
                        value = seasonal_value
                    else:
                        value = 0.80 * seasonal_value + 0.20 * recent_value
                else:
                    value = seasonal_value
            elif not recent.empty:
                value = float(recent.median())
            else:
                value = float(y.median())
            values.append(max(value, 0.0))
        return pd.Series(values, index=clean_series.index)

    @staticmethod
    def _build_ml_features(series: pd.Series, dates: pd.Series = None) -> pd.DataFrame:
        """构造严格因果特征：每一天的特征只使用当天之前的 cleaned history。"""
        y = pd.Series(series).astype(float).reset_index(drop=True)
        n = len(y)
        t = np.arange(n, dtype=float)

        if dates is not None:
            day_of_week = pd.to_datetime(dates).reset_index(drop=True).dt.dayofweek.astype(float)
        else:
            day_of_week = pd.Series(t % 7)

        features = pd.DataFrame({
            't': t,
            'dow_sin': np.sin(2 * np.pi * day_of_week / 7.0),
            'dow_cos': np.cos(2 * np.pi * day_of_week / 7.0),
            'lag_1': y.shift(1),
            'lag_7': y.shift(7),
            'lag_14': y.shift(14),
            'lag_21': y.shift(21),
            'rolling_median_7': y.shift(1).rolling(7, min_periods=1).median(),
            'rolling_median_14': y.shift(1).rolling(14, min_periods=1).median(),
            'rolling_median_21': y.shift(1).rolling(21, min_periods=1).median(),
            'rolling_mean_7': y.shift(1).rolling(7, min_periods=1).mean(),
            'same_weekday_median_4': pd.concat(
                [y.shift(7), y.shift(14), y.shift(21), y.shift(28)], axis=1
            ).median(axis=1),
        })

        causal_fill = y.shift(1).expanding(min_periods=1).median().bfill().fillna(y.median())
        features = features.apply(lambda col: col.fillna(causal_fill))
        return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _historical_inlier_mask(
        y_train: pd.Series, trend_window: int, threshold_multiplier: float, min_deviation: float
    ) -> np.ndarray:
        """
        用滚动中位数 + MAD 先剔除历史中的明显异常点，避免模型学习异常。
        返回 True 表示该历史点可参与训练。
        """
        if len(y_train) < 8:
            return np.ones(len(y_train), dtype=bool)

        baseline = y_train.shift(1).rolling(trend_window, min_periods=1).median()
        baseline = baseline.bfill().fillna(y_train.median())
        residual = y_train - baseline
        med = residual.median()
        mad = np.median(np.abs(residual - med))
        robust_scale = max(mad * 1.4826, residual.std() if pd.notna(residual.std()) else 0.0, 1.0)
        allowed = np.maximum(min_deviation, baseline * 0.30)
        mask = (np.abs(residual - med) <= threshold_multiplier * robust_scale) | (np.abs(residual) <= allowed)

        # 避免极端情况下过滤过多，保留大部分数据的规律作为训练主体。
        if mask.sum() < max(8, int(len(y_train) * 0.55)):
            return np.ones(len(y_train), dtype=bool)
        return mask.to_numpy(dtype=bool)

    @staticmethod
    def _run_sklearn_predictive_anomaly(
        series: pd.Series, trend_window: int, history_window: int,
        threshold_multiplier: float, min_deviation: float, dates: pd.Series = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        sklearn 鲁棒预测内核：
        1. 每一天只用之前的历史训练，避免未来数据穿越。
        2. 训练前用滚动中位数/MAD 剔除历史异常点。
        3. 使用 HuberRegressor 拟合大多数数据的趋势、周周期、滞后和滚动特征。
        """
        if not HAS_SKLEARN:
            raise ImportError("volume_spike 检测需要 scikit-learn，请先安装: pip install scikit-learn")

        original_index = series.index
        raw_y = pd.Series(series).astype(float).reset_index(drop=True)
        clean_y = AnomalyDetectorEngine._build_causal_clean_series(
            raw_y, trend_window, history_window, threshold_multiplier, min_deviation
        )
        features = AnomalyDetectorEngine._build_ml_features(clean_y, dates)
        weekly_baseline = AnomalyDetectorEngine._causal_weekly_baseline(clean_y, trend_window)
        expected_values = []
        trained_inlier_counts = []
        min_train_points = 14

        for i in range(len(raw_y)):
            if i < min_train_points:
                fallback = weekly_baseline.iloc[i] if i > 0 else clean_y.median()
                expected_values.append(max(float(fallback), 0.0))
                trained_inlier_counts.append(i)
                continue

            y_train = clean_y.iloc[:i]
            x_train = features.iloc[:i]
            x_pred = features.iloc[[i]]
            inlier_mask = AnomalyDetectorEngine._historical_inlier_mask(
                y_train, trend_window, threshold_multiplier, min_deviation
            )

            if inlier_mask.sum() < min_train_points:
                fallback = y_train.tail(trend_window).median()
                expected_values.append(max(float(fallback), 0.0))
                trained_inlier_counts.append(int(inlier_mask.sum()))
                continue

            model = make_pipeline(
                StandardScaler(),
                HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)
            )
            try:
                model.fit(x_train.loc[inlier_mask], y_train.loc[inlier_mask])
                pred = float(model.predict(x_pred)[0])
            except Exception:
                pred = float(y_train.loc[inlier_mask].tail(trend_window).median())

            seasonal_pred = float(weekly_baseline.iloc[i])
            if i >= 28:
                pred = 0.45 * pred + 0.55 * seasonal_pred
            elif i >= 14:
                pred = 0.65 * pred + 0.35 * seasonal_pred

            expected_values.append(max(pred, 0.0))
            trained_inlier_counts.append(int(inlier_mask.sum()))

        expected_curve = pd.Series(expected_values, index=original_index).clip(lower=0)
        deviation = series - expected_curve
        abs_deviation = np.abs(deviation)
        clean_residual = pd.Series(clean_y.values, index=original_index) - expected_curve

        def calc_mad(x):
            if len(x) == 0: return 0.0
            return np.median(np.abs(x - np.median(x)))

        rolling_mad = clean_residual.shift(1).rolling(window=history_window, min_periods=5).apply(calc_mad, raw=True)
        rolling_std = clean_residual.shift(1).rolling(window=history_window, min_periods=5).std()

        scale = rolling_mad * 1.4826
        scale = np.where(scale < 1e-5, rolling_std, scale)
        dynamic_scale = pd.Series(scale, index=original_index).bfill().fillna(1.0).clip(lower=1.0)

        dynamic_min = np.maximum(min_deviation, expected_curve * 0.30)

        anomaly_score = abs_deviation / dynamic_scale
        raw_width = dynamic_scale * threshold_multiplier
        width_cap = np.maximum(min_deviation * 3.0, expected_curve * 0.75)
        threshold_width = np.maximum(dynamic_min, np.minimum(raw_width, width_cap))
        anomaly_score = abs_deviation / (threshold_width / max(threshold_multiplier, 1e-6)).clip(lower=1.0)
        is_anomaly = (abs_deviation > threshold_width)

        upper_bound = expected_curve + threshold_width
        lower_bound = (expected_curve - threshold_width).clip(lower=0)

        # 保存下界和训练信息，供结构化输出与 CSV 调试使用。
        expected_curve.attrs['model_name'] = 'sklearn_huber_regressor'
        expected_curve.attrs['trained_inlier_counts'] = trained_inlier_counts
        expected_curve.attrs['lower_bound'] = lower_bound
        expected_curve.attrs['cleaned_series'] = pd.Series(clean_y.values, index=original_index)

        return is_anomaly, anomaly_score, expected_curve, dynamic_scale, upper_bound

    # ------------------- 模式 1: 单项及总量因果突增检测 -------------------
    @staticmethod
    def run_volume_spike_sliding(
        df: pd.DataFrame, target_cols: List[str], 
        threshold_multiplier: float, min_deviation: float, 
        trend_window: int, history_window: int
    ) -> pd.DataFrame:
        df = df.copy()
        df['total'] = df[target_cols].sum(axis=1)
        
        # 1. 扫描大盘总量
        tot_anom, tot_score, exp_tot, sc_tot, ub_tot = AnomalyDetectorEngine._run_sklearn_predictive_anomaly(
            df['total'], trend_window, history_window, threshold_multiplier, min_deviation, df['event_date']
        )
        df['expected_total'] = exp_tot
        df['total_lower_bound'] = exp_tot.attrs.get('lower_bound', pd.Series(0.0, index=df.index))
        df['total_upper_bound'] = ub_tot
        df['is_total_anomaly'] = tot_anom
        df['total_ml_train_inliers'] = exp_tot.attrs.get('trained_inlier_counts', [0] * len(df))
        df['total_cleaned_for_model'] = exp_tot.attrs.get('cleaned_series', df['total'])
        
        # 2. 扫描所有独立单项
        any_sub_anom = np.zeros(len(df), dtype=bool)
        anom_reasons = [''] * len(df)
        max_scores = tot_score.copy()
        
        sub_results = {}
        for col in target_cols:
            if df[col].max() < 1: continue
            sub_anom, sub_score, sub_exp, _, sub_ub = AnomalyDetectorEngine._run_sklearn_predictive_anomaly(
                df[col], trend_window, history_window, threshold_multiplier, min_deviation, df['event_date']
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
    def plot_volume_spike_sliding(df: pd.DataFrame, fleet_id: str, save_path: str):
        """
        生成总量与单项突增智能可视化面板。
        如果有单项指标异常，会自动拆分出专属的子图，防止被大盘量级掩蔽。
        """
        dates = df['event_date']
        
        # 提取所有发生过异常的单项事件
        anomalous_items = [c.replace('sub_anom_', '') for c in df.columns if c.startswith('sub_anom_')]
        
        # 为了防止图表过长，如果异常单项过多，按异常严重度截取前 4 个最显著的
        if anomalous_items:
            item_max_scores = {col: df[f'sub_score_{col}'].max() for col in anomalous_items}
            sorted_items = sorted(item_max_scores.keys(), key=lambda k: item_max_scores[k], reverse=True)
            top_items = sorted_items[:4] 
        else:
            top_items = []
            
        num_plots = 1 + len(top_items)
        fig = plt.figure(figsize=(16, 6 * num_plots), constrained_layout=True)
        gs = plt.GridSpec(num_plots, 1)
        
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
        
        # --- Plot 2~N: 独立分离出的发生激增的单项图 ---
        for idx, col in enumerate(top_items):
            ax_sub = fig.add_subplot(gs[idx + 1])
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
                                
            ax_sub.set_title(f'{idx + 2}. Individual Item Spike: {col}', fontsize=14, fontweight='bold', color='darkred')
            ax_sub.set_ylabel('Event Count', fontsize=12)
            ax_sub.grid(True, alpha=0.3)
            ax_sub.legend(loc='upper right', fontsize=11)
            
        fig.suptitle(f'Continuous Event Analysis - Fleet: {fleet_id}', fontsize=18, fontweight='bold', y=1.02 if num_plots == 1 else 1.0)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ------------------- 模式 2: 分布占比连续滑动检测 (JS 散度逻辑) -------------------
    @staticmethod
    def run_distribution_shift_sliding(
        df: pd.DataFrame, target_cols: List[str], 
        threshold_multiplier: float, min_divergence: float, trend_window: int, history_window: int
    ) -> pd.DataFrame:
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
        """生成分布漂移回测可视化大图"""
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


# ==========================================
# 2. 定义 Tool
# ==========================================
class FleetAnomalyDetectionTool(BaseTool):
    name: str = "fleet_behavior_anomaly_detector"
    description: str = (
        "扫描指定时间段（可长达一年）的车队告警数据，提供每日维度的宏观异常检验并生成全景可视化图表。"
        "模式支持：结构占比漂移崩塌 'distribution_shift'，总及单项数据突发激增 'volume_spike'。"
    )
    
    env_path: str = "./config/env.json" 
    
    def get_db_connection(self):
        """加载环境配置获取数据库连接"""
        if not os.path.exists(self.env_path):
            raise FileNotFoundError(f"配置文件不存在: {self.env_path}")
        with open(self.env_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        conn_params = config['conn_params']
        conn = psycopg2.connect(
            host=conn_params['host'], database=conn_params['database'],
            user=conn_params['user'], password=conn_params['password'], port=conn_params['port']
        )
        return conn

    def _fetch_fleet_data(self, fleet_id: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[str]]:
        print(f"[Tool Backend] Fetching DB for fleet: {fleet_id} | {start_date} to {end_date}")
        conn = self.get_db_connection()
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

        if df_raw.empty: return pd.DataFrame(), []
        df_pivoted = df_raw.pivot_table(index=['fleetid', 'event_date'], columns='eventtype', values='event_count', fill_value=0).reset_index()
        df_pivoted['event_date'] = pd.to_datetime(df_pivoted['event_date'])
        df_pivoted = df_pivoted.sort_values('event_date').reset_index(drop=True)
        target_columns = [col for col in df_pivoted.columns if col not in ['fleetid', 'event_date']]
        return df_pivoted, target_columns

    def _run(
        self, fleet_id: str, start_date: str, end_date: str, 
        detection_mode: str = "all",
        enable_visualization: bool = False,
        threshold_multiplier: float = 3.5, 
        min_abs_deviation: float = 50.0, 
        min_deviation: float = 0.25, 
        trend_window: int = 21, 
        history_window: int = 30
    ) -> str:
        try:
            try:
                df, target_columns = self._fetch_fleet_data(fleet_id, start_date, end_date)
            except:
                import traceback
                traceback.print_exc()
                print("⚠️ 数据库连接失败或配置文件缺失，进入 MOCK 测试模式。")
                return self._mock_run_for_testing(
                    fleet_id, start_date, end_date, detection_mode, enable_visualization, threshold_multiplier, 
                    min_abs_deviation, min_deviation, trend_window, history_window
                )

            if df.empty: return f"⚠️ 无法检测：指定时间段内，车队 {fleet_id} 没有任何数据。"
            if not target_columns: return f"⚠️ 字段异常：未发现任何事件字段。"

            out_dir = './agent_workspace'
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M')

            modes = self._resolve_detection_modes(detection_mode)
            if not modes:
                return f"⚠️ 未知的检测模式: {detection_mode}。仅支持 'all'、'distribution_shift' 或 'volume_spike'。"

            mode_results = []
            for mode in modes:
                res_df, csv_path, plot_path = self._run_one_mode(
                    mode, df, target_columns, fleet_id, timestamp, out_dir,
                    threshold_multiplier, min_abs_deviation, min_deviation,
                    trend_window, history_window, enable_visualization
                )
                mode_results.append((mode, res_df, csv_path, plot_path))

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

    def _mock_run_for_testing(
        self, fleet_id, start_date, end_date, detection_mode, enable_visualization,
        th_mult, min_abs_dev, min_dev, trend_win, hist_win
    ):
        df = pd.read_csv('./data/all_fleets_timeseries.csv')
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df[df['fleetid'] == fleet_id]
        mask = (df['event_date'] >= pd.to_datetime(start_date)) & (df['event_date'] <= pd.to_datetime(end_date))
        df = df.loc[mask].sort_values('event_date').reset_index(drop=True)
        target_columns = [col for col in df.columns if col not in ['fleetid', 'event_date', 'fleetname', 'total_events']]
        if df.empty:
            return f"⚠️ MOCK 无法检测：指定时间段内，车队 {fleet_id} 没有任何数据。"
        if not target_columns:
            return f"⚠️ MOCK 字段异常：未发现任何事件字段。"
        
        out_dir = './agent_workspace'
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M')

        modes = self._resolve_detection_modes(detection_mode)
        if not modes:
            return f"⚠️ 未知的检测模式: {detection_mode}。仅支持 'all'、'distribution_shift' 或 'volume_spike'。"

        mode_results = []
        for mode in modes:
            res_df, csv_path, plot_path = self._run_one_mode(
                mode, df, target_columns, fleet_id, f"mock_{timestamp}", out_dir,
                th_mult, min_abs_dev, min_dev, trend_win, hist_win, enable_visualization
            )
            mode_results.append((mode, res_df, csv_path, plot_path))

        if not mode_results or mode_results[0][1].empty:
            return f"⚠️ MOCK 无法检测：指定时间段内，车队 {fleet_id} 没有任何数据。"

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

    @staticmethod
    def _clean_number(value: Any):
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _build_structured_result(
        fleet_id: str, start_date: str, end_date: str, last_event_date: pd.Timestamp,
        modes: List[str], total_days: int, anomaly_records: List[Dict[str, Any]],
        summary_path: str
    ) -> Dict[str, Any]:
        return {
            "summary": {
                "fleet_id": fleet_id,
                "event_window": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "judgement_date": last_event_date.strftime('%Y-%m-%d')
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
                "judgement_date": event_date.strftime('%Y-%m-%d')
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
                            "model": "sklearn_huber_regressor",
                            "trained_inlier_count": cls._clean_number(last_row.get('total_ml_train_inliers', None))
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
                            "model": "sklearn_huber_regressor",
                            "trained_inlier_count": cls._clean_number(last_row.get(f'sub_ml_train_inliers_{col}', None))
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

        return records

    @staticmethod
    def _resolve_detection_modes(detection_mode: str) -> List[str]:
        mode = (detection_mode or "all").strip().lower()
        if mode in {"all", "both", "全部", "所有"}:
            return ["distribution_shift", "volume_spike"]
        if mode in {"distribution_shift", "volume_spike"}:
            return [mode]
        return []

    @staticmethod
    def _run_one_mode(
        mode: str, df: pd.DataFrame, target_columns: List[str], fleet_id: str,
        timestamp: str, out_dir: str, threshold_multiplier: float,
        min_abs_deviation: float, min_deviation: float, trend_window: int,
        history_window: int, enable_visualization: bool
    ) -> Tuple[pd.DataFrame, Optional[str], str]:
        if mode == "distribution_shift":
            res_df = AnomalyDetectorEngine.run_distribution_shift_sliding(
                df, target_columns, threshold_multiplier, min_deviation, trend_window, history_window
            )
            plot_path = f"{out_dir}/distribution_curve_{fleet_id}_{timestamp}.png"
            if enable_visualization:
                AnomalyDetectorEngine.plot_distribution_shift_sliding(res_df, target_columns, fleet_id, plot_path)
        elif mode == "volume_spike":
            res_df = AnomalyDetectorEngine.run_volume_spike_sliding(
                df, target_columns, threshold_multiplier, min_abs_deviation, trend_window, history_window
            )
            plot_path = f"{out_dir}/volume_curve_{fleet_id}_{timestamp}.png"
            if enable_visualization:
                AnomalyDetectorEngine.plot_volume_spike_sliding(res_df, fleet_id, plot_path)
        else:
            raise ValueError(f"Unsupported detection mode: {mode}")

        csv_path = None
        if enable_visualization:
            csv_path = f"{out_dir}/scan_data_{mode}_{fleet_id}_{timestamp}.csv"
            res_df.to_csv(csv_path, index=False)
        return res_df, csv_path, plot_path


# ==========================================
# 4.  调用
# ==========================================
def test_agent_tool():
    anomaly_tool = FleetAnomalyDetectionTool()
    time_window_days = DEFAULT_TIME_WINDOW_DAYS
    date_window_args = build_time_window_params(time_window_days)
    
    agent_args = {
        "fleet_id": "c6329dff7db740a2848b9d34ca6bd7af",
        "start_date": date_window_args["start_date"],
        "end_date": date_window_args["end_date"],
        "detection_mode": "all", 
        "enable_visualization": True,
        "threshold_multiplier": 7.5,
        "trend_window": 90,
        "history_window": 90         
    }
    print("\n▶️ 测试：全模式事件窗口末日异常检测")
    print(anomaly_tool._run(**agent_args))

if __name__ == "__main__":
    test_agent_tool()
