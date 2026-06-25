# 异常检测结构化结果说明

本文档只说明异常检测返回结果的通用结构化模板，不依赖任何一次具体运行结果。将检测结果 JSON 交给 AI/LLM 时，可以同时提供本文档，帮助 AI 理解字段含义、数据类型和可能取值。

## 顶层结构

多模式执行时，最终返回结果是一个 list。list 中每个元素对应一个检测模式的一次原始结构化返回。

```json
[
  {
    "summary": {},
    "anomalies": []
  }
]
```

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `[]` | `array<object>` | 是 | 多模式结果列表。每个元素是一个模式的原始结构化结果。 |
| `[].summary` | `object` | 是 | 单个检测模式的摘要信息。 |
| `[].anomalies` | `array<object>` | 是 | 单个检测模式在判定日收集到的异常列表。没有异常时为空数组。 |

## summary 模板

```json
{
  "fleet_id": "string",
  "event_window": {
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "judgement_date": "YYYY-MM-DD"
  },
  "detection_modes": ["volume_spike"],
  "total_days": 90,
  "has_anomaly": true,
  "anomaly_count": 1,
  "summary_file": "agent_workspace/last_day_anomaly_summary_..."
}
```

### summary 字段说明

| 字段 | 类型 | 必填 | 可能取值 | 说明 |
|---|---|---:|---|---|
| `fleet_id` | `string` | 是 | 任意车队 ID 或数据源名称 | 当前检测结果对应的车队或数据源。 |
| `event_window` | `object` | 是 | 固定对象 | 当前检测模式实际使用的时间窗口。 |
| `event_window.start_date` | `string` | 是 | `YYYY-MM-DD` | 检测窗口开始日期。 |
| `event_window.end_date` | `string` | 是 | `YYYY-MM-DD` | 检测窗口结束日期。 |
| `event_window.judgement_date` | `string` | 是 | `YYYY-MM-DD` | 最终判定日期，通常是窗口内最后一天。 |
| `detection_modes` | `array<string>` | 是 | 见“检测模式取值” | 当前结果对应的检测模式。单模式调用时通常只有一个元素。 |
| `total_days` | `integer` | 是 | `>= 0` | 当前模式实际参与检测的数据天数。 |
| `has_anomaly` | `boolean` | 是 | `true`、`false` | 当前模式在判定日是否检测到异常。 |
| `anomaly_count` | `integer` | 是 | `>= 0` | 当前模式在判定日收集到的异常条数。 |
| `summary_file` | `string` | 是 | 文件路径字符串 | 当前模式单独保存的结构化结果文件路径。 |

## anomalies 模板

```json
{
  "fleet_id": "string",
  "event_window": {
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "judgement_date": "YYYY-MM-DD"
  },
  "detection_mode": "volume_spike",
  "eventtype": "__total__",
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
  "reason": "异常原因文本",
  "extra": {}
}
```

### anomalies 字段说明

| 字段 | 类型 | 必填 | 可能取值 | 说明 |
|---|---|---:|---|---|
| `fleet_id` | `string` | 是 | 任意车队 ID 或数据源名称 | 异常所属车队或数据源。 |
| `event_window` | `object` | 是 | 固定对象 | 异常所属模式使用的时间窗口。 |
| `event_window.start_date` | `string` | 是 | `YYYY-MM-DD` | 检测窗口开始日期。 |
| `event_window.end_date` | `string` | 是 | `YYYY-MM-DD` | 检测窗口结束日期。 |
| `event_window.judgement_date` | `string` | 是 | `YYYY-MM-DD` | 该异常的判定日期。 |
| `detection_mode` | `string` | 是 | 见“检测模式取值” | 触发该异常的检测模式。 |
| `eventtype` | `string` | 是 | `__total__`、事件类型名、指标名 | 异常对应的事件类型。总量异常通常为 `__total__`。 |
| `metric` | `string` | 是 | 见“指标取值” | 异常对应的指标类型。 |
| `actual_value` | `number|null` | 是 | 数值或 `null` | 判定日真实值。 |
| `expected_value` | `number|null` | 是 | 数值或 `null` | 基于判定日前历史数据得到的期望值。 |
| `threshold_interval` | `object` | 是 | 固定对象 | 异常判断阈值区间。真实值在区间内通常视为正常。 |
| `threshold_interval.lower` | `number|null` | 是 | 数值或 `null` | 正常区间下界。 |
| `threshold_interval.upper` | `number|null` | 是 | 数值或 `null` | 正常区间上界。 |
| `direction` | `string` | 是 | 见“异常方向取值” | 真实值相对期望值或阈值区间的方向。 |
| `is_anomaly` | `boolean` | 是 | `true`、`false` | 当前记录是否为异常。正常情况下 `anomalies` 列表内记录应为 `true`。 |
| `anomaly_score` | `number|null` | 是 | 数值或 `null` | 异常严重度评分。数值越大通常代表偏离越严重。 |
| `reason` | `string` | 是 | 任意文本 | 可读的异常诊断原因。 |
| `extra` | `object` | 是 | 模式相关对象 | 检测模式补充信息。不同模式字段可能不同。 |

## 检测模式取值

| 取值 | 类型 | 含义 |
|---|---|---|
| `volume_spike` | `string` | 事件总量或单项事件数量相对历史趋势异常升高或降低。 |
| `distribution_shift` | `string` | 事件类型占比结构相对历史分布发生异常漂移。 |
| `events_per_hour` | `string` | 每小时事件数指标相对历史趋势异常。 |

工具内部可能支持 `all` 作为调用参数，但多模式结果中建议保留每个实际执行模式的原始结果，而不是把 `all` 作为分析模式。

## 指标取值

| 取值 | 类型 | 含义 |
|---|---|---|
| `event_count` | `string` | 事件数量。常见于 `volume_spike`。 |
| `event_proportion` | `string` | 某事件类型占总事件量的比例。常见于 `distribution_shift`。 |
| `events_per_hour` | `string` | 每小时事件数。常见于 `events_per_hour`。 |

如果后续新增检测模式，`metric` 可能出现新的字符串。AI 应优先根据 `detection_mode`、`reason` 和 `extra` 理解其含义。

## 异常方向取值

| 取值 | 类型 | 含义 |
|---|---|---|
| `too_high` | `string` | 真实值高于期望值或高于阈值上界。 |
| `too_low` | `string` | 真实值低于期望值或低于阈值下界。 |
| `equal_to_expected` | `string` | 真实值与期望值基本一致。 |
| `unknown` | `string` | 无法明确判断方向，通常表示缺少期望值或阈值信息。 |

## extra 常见字段

`extra` 是模式扩展字段，不同检测模式可能不同。AI 不应假设它固定包含所有字段。

| 字段 | 类型 | 可能取值 | 说明 |
|---|---|---|---|
| `scope` | `string` | `total_volume`、`single_eventtype`、其他字符串 | 异常作用范围。 |
| `model` | `string` | 任意模型名 | 生成期望值或阈值所用的模型/算法说明。 |
| `period` | `integer` | `>= 1` | 周期长度，例如 7 表示按 7 天周期相位分析。 |
| `phase` | `integer|null` | 数值或 `null` | 周期相位。 |
| `baseline_count` | `integer|null` | 数值或 `null` | 历史基线样本数量。 |

## AI 解读建议

1. 顶层 list 中每个元素独立代表一个检测模式的结果。
2. 判断整体是否异常时，可以检查任意元素的 `summary.has_anomaly` 是否为 `true`。
3. 解释异常时，优先读取各元素的 `anomalies` 列表。
4. 不要把不同模式的 `summary` 强行合并，因为不同模式可能使用不同时间窗口、不同数据源和不同指标。
5. 如果 `anomalies` 为空数组，表示该模式在判定日没有收集到异常。
6. `expected_value` 和 `threshold_interval` 都只应理解为基于判定日前历史数据得到的检测参考，不代表使用未来数据。
