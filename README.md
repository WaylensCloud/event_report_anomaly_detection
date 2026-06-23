# event_report_anomaly_detection
event_report界面数据异常检测
# Fleet Behavior Anomaly Detector

`ad_tool.py` 是一个面向 Agent/LLM 工具调用的车队事件时间序列异常检测工具。它会扫描指定车队在一个事件窗口内的每日事件数据，并只返回窗口最后一天是否异常以及对应的结构化异常信息。

## 主要用途

该工具用于回答类似问题：

- 某个车队最近一段时间最后一天是否出现异常事件？
- 异常来自总量突增、单项事件突增，还是事件结构分布漂移？
- 异常事件的真实值、历史期望值、阈值区间、异常方向是什么？
- 如何把异常检测结果以固定 JSON 格式交给后续 LLM 继续分析？

## 检测模式

工具支持三种 `detection_mode`：

- `all`：默认模式，同时运行所有检测模式。
- `volume_spike`：检测总事件量或单个 `eventtype` 是否相对历史期望异常升高或降低。
- `distribution_shift`：检测各类事件占比结构是否发生异常漂移。

## 时间窗口

推荐在外层先根据 `time_window_days` 计算最终日期参数，再传入工具：

```python
from agent_vis import (
    FleetAnomalyDetectionTool,
    DEFAULT_TIME_WINDOW_DAYS,
    build_time_window_params,
)

tool = FleetAnomalyDetectionTool()

date_window_args = build_time_window_params(DEFAULT_TIME_WINDOW_DAYS)

result = tool._run(
    fleet_id="your_fleet_id",
    **date_window_args,
    detection_mode="all",
    enable_visualization=False,
)
```

`build_time_window_params()` 会生成：

```python
{
    "time_window_days": 90,
    "start_date": "今天 - 90 天",
    "end_date": "今天"
}
```

`_run()` 本身不会再重新计算日期，它只使用传入的最终 `start_date` 和 `end_date`。

## 可视化和 CSV

`enable_visualization` 控制额外文件输出：

- `False`：不保存图表，不保存 `scan_data_*.csv`，只保存末日异常汇总 JSON。
- `True`：保存图表，并保存每个检测模式对应的详细 CSV 底表。

末日异常汇总 JSON 始终保存，因为它是给后续程序或 LLM 使用的核心输出。

## 返回给后续 LLM 的结构

工具返回文本最后会包含固定标记：

```text
STRUCTURED_ANOMALY_RESULT_JSON:
{
  "summary": {...},
  "anomalies": [...]
}
```

后续 LLM 可以直接定位 `STRUCTURED_ANOMALY_RESULT_JSON:` 后面的 JSON，并解析 `summary` 与 `anomalies`。

### summary 字段

```json
{
  "fleet_id": "your_fleet_id",
  "event_window": {
    "start_date": "2026-03-25",
    "end_date": "2026-06-23",
    "judgement_date": "2026-06-23"
  },
  "detection_modes": ["distribution_shift", "volume_spike"],
  "total_days": 90,
  "has_anomaly": true,
  "anomaly_count": 3,
  "summary_file": "./agent_workspace/last_day_anomaly_summary_..."
}
```

### anomalies 字段

`anomalies` 是一个 list。每条异常使用固定格式：

```json
{
  "fleet_id": "your_fleet_id",
  "event_window": {
    "start_date": "2026-03-25",
    "end_date": "2026-06-23",
    "judgement_date": "2026-06-23"
  },
  "detection_mode": "volume_spike",
  "eventtype": "harsh_brake",
  "metric": "event_count",
  "actual_value": 120.0,
  "expected_value": 45.0,
  "threshold_interval": {
    "lower": 10.0,
    "upper": 80.0
  },
  "direction": "too_high",
  "is_anomaly": true,
  "anomaly_score": 6.3,
  "reason": "harsh_brake 计数异常",
  "extra": {
    "scope": "single_eventtype"
  }
}
```

字段说明：

- `detection_mode`：触发异常的检测模式。
- `eventtype`：异常对应的事件类型。总量异常使用 `__total__`。
- `metric`：指标类型，例如 `event_count` 或 `event_proportion`。
- `actual_value`：最后一天真实值。
- `expected_value`：基于历史数据得到的期望值。
- `threshold_interval`：异常判断阈值区间。
- `direction`：相对历史期望是 `too_high`、`too_low`、`equal_to_expected` 或 `unknown`。
- `anomaly_score`：异常严重度评分。
- `reason`：可读诊断原因。
- `extra`：模式相关补充信息。

## 数据来源

默认真实数据来自 PostgreSQL 视图 `v_clip_wide_api`，连接配置从：

```text
./config/env.json
```

读取。如果数据库配置缺失或连接失败，代码会进入 mock 测试分支，读取：

```text
./output/all_fleets_timeseries.csv
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 文件说明

- `agent_vis.py`：核心异常检测工具。
- `requirements.txt`：运行依赖。
- `README.md`：使用说明和返回格式说明。
