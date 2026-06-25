# event_report_anomaly_detection

车队事件异常检测工具。当前主入口是模块化包 `anomaly_detection`，用于扫描指定时间窗口内的事件数据，并重点输出事件窗口最后一天的结构化异常信息，方便后续 LLM 或程序继续分析。

## 目录结构

```text
event_report_anomaly_detection/
├── anomaly_detection/
│   ├── config/
│   │   ├── constants.py          # 默认时间窗口与日期工具
│   │   └── settings.py           # 配置读取与数据库连接
│   ├── core/
│   │   └── engine.py             # 异常检测算法与可视化
│   │   └── engine_clean.py       # 当前默认使用的干净状态机版趋势检测
│   ├── fetchers/
│   │   ├── fleet_data.py         # 车队事件数据获取
│   │   └── events_per_hour.py    # events_per_hour 数据获取
│   └── tool.py                   # Agent/LLM 工具入口
├── agent_workspace/              # 输出文件目录
├── config/env.json               # 数据库配置
├── execute.py                    # 多模式循环调用示例
├── RESULT_SCHEMA.md              # 返回结果结构说明，供 AI/LLM 理解结果
└── requirements.txt
```

## 支持模式

- `distribution_shift`：检测事件类型占比结构是否发生异常漂移。
- `volume_spike`：检测总事件量和单项事件类型是否相对历史趋势异常升高或降低。
- `events_per_hour`：检测每小时事件数指标是否异常。
- `all`：工具内部仍支持，但示例代码不使用 `all`。推荐像 `execute.py` 一样按模式循环调用，便于每种模式使用不同参数。

## 快速运行

```bash
pip install -r requirements.txt
python execute.py
```

默认会分别运行 `volume_spike`、`events_per_hour`，不会调用 `detection_mode="all"`。

也可以只运行指定模式：

```bash
python execute.py volume_spike
python execute.py distribution_shift events_per_hour
```

## 示例代码逻辑

`execute.py` 中的 `build_mode_args()` 为每种模式单独配置参数，例如：

- `volume_spike` 使用 `spike_period=7`，按 7 天周期相位分析。
- `events_per_hour` 使用 `spike_period=1`，因为它是连续小数指标。
- 不同模式可以设置不同的 `min_abs_deviation`、`expected_relative_tolerance`、`threshold_multiplier` 等参数。
- `END_DATE` 是最外层结束日期配置。默认为 `None` 时使用今天；设置为 `"YYYY-MM-DD"` 时，所有模式都会以该日期作为 `end_date`，再按各自窗口天数动态计算 `start_date`。

每个模式运行后，工具返回文本末尾都会带有：

```text
STRUCTURED_ANOMALY_RESULT_JSON:
{
  "summary": {...},
  "anomalies": [...]
}
```

`execute.py` 会解析每个模式的这段 JSON，并把所有模式的原始结构化结果直接拼成一个 list，不额外包装、不打散、不改字段：

```json
[
  {
    "summary": {
      "fleet_id": "c6329dff7db740a2848b9d34ca6bd7af",
      "event_window": {
        "start_date": "2026-03-27",
        "end_date": "2026-06-25",
        "judgement_date": "2026-06-25"
      },
      "detection_modes": ["volume_spike"],
      "total_days": 90,
      "has_anomaly": true,
      "anomaly_count": 1,
      "summary_file": "agent_workspace/last_day_anomaly_summary_..."
    },
    "anomalies": []
  },
  {
    "summary": {
      "fleet_id": "GroundCloud",
      "event_window": {
        "start_date": "2026-05-26",
        "end_date": "2026-06-25",
        "judgement_date": "2026-06-25"
      },
      "detection_modes": ["events_per_hour"],
      "total_days": 30,
      "has_anomaly": false,
      "anomaly_count": 0,
      "summary_file": "agent_workspace/last_day_anomaly_summary_..."
    },
    "anomalies": []
  }
]
```

多模式结果文件会保存到：

```text
agent_workspace/multi_mode_anomaly_results_{fleet_id}_{timestamp}.json
```

这个文件顶层数据类型是 `array<object>`。数组中每个元素都保持单个模式原始返回结构：

- `summary`：`object`，单个模式的摘要。
- `anomalies`：`array<object>`，该模式在判定日检测到的异常列表；没有异常时为空数组。

### summary 字段说明

- `fleet_id`：`string`，车队 ID 或该模式使用的数据源名称。
- `event_window`：`object`，该模式实际使用的检测时间窗口。
- `event_window.start_date`：`string`，格式 `YYYY-MM-DD`，检测窗口开始日期。
- `event_window.end_date`：`string`，格式 `YYYY-MM-DD`，检测窗口结束日期。
- `event_window.judgement_date`：`string`，格式 `YYYY-MM-DD`，最终判定日期，通常是窗口内最后一天。
- `detection_modes`：`array<string>`，该结果对应的检测模式。按模式单独调用时通常只有一个元素。
- `total_days`：`integer`，该模式实际参与检测的天数。
- `has_anomaly`：`boolean`，该模式判定日是否存在异常。
- `anomaly_count`：`integer`，该模式判定日异常条数。
- `summary_file`：`string`，该模式单独保存的结构化结果文件路径。

## 单条异常格式

`anomalies` 是一个 list。每条异常使用固定结构：

```json
{
  "fleet_id": "your_fleet_id",
  "event_window": {
    "start_date": "2026-03-27",
    "end_date": "2026-06-25",
    "judgement_date": "2026-06-25"
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
  "reason": "总事件量异常",
  "extra": {
    "scope": "total_volume"
  }
}
```

关键字段：

- `fleet_id`：`string`，车队 ID。
- `event_window`：`object`，该异常所属模式的时间窗口。
- `detection_mode`：`string`，触发异常的检测模式。
- `eventtype`：`string`，异常事件类型。总量异常使用 `__total__`。
- `metric`：`string`，指标名，例如 `event_count`、`event_proportion`、`events_per_hour`。
- `actual_value`：`number|null`，判定日真实值。
- `expected_value`：`number|null`，只基于判定日之前历史数据得到的期望值。
- `threshold_interval`：`object`，safe range，通常包含 `lower` 和 `upper`。
- `threshold_interval.lower`：`number|null`，正常区间下界。
- `threshold_interval.upper`：`number|null`，正常区间上界。
- `direction`：`string`，`too_high`、`too_low`、`equal_to_expected` 或 `unknown`。
- `is_anomaly`：`boolean`，该记录是否为异常。
- `anomaly_score`：`number|null`，异常严重度评分。
- `reason`：`string`，可读诊断原因。
- `extra`：`object`，模式相关补充字段。

## 主要参数

- `start_date` / `end_date`：外部传入的最终时间窗口，工具内部不重新计算日期。
- `enable_visualization`：`False` 时不保存图表和 CSV，只保存结构化 JSON；`True` 时额外保存底表和可视化图。
- `spike_period`：强制周期长度。`7` 表示按星期相位拆分历史，`1` 表示不拆相位。
- `stable_regime_points`：连续多少个同相位点稳定后，允许预测趋势切换到新水平。
- `stable_regime_tolerance_ratio`：判断新趋势是否稳定时允许的相对波动。
- `stable_shift_min_ratio`：新趋势相对旧趋势至少变化多少，才认为可能发生稳定台阶切换。
- `expected_relative_tolerance`：safe range 相对预测值的基础百分比容忍度。
- `threshold_width_cap_ratio`：safe range 最大宽度相对预测值的上限倍数。
- `lower_anomaly_tolerance_multiplier`：低于预测值时额外放宽倍数，用于降低周期低谷误报。
- `normal_dispersion_multiplier`：历史正常离散度放大倍数。
- `normal_dispersion_floor_ratio`：历史正常离散度的相对下限。
- `normal_dispersion_min_points`：至少多少个历史正常点后启用离散度宽度。
- `threshold_multiplier`：MAD/尺度阈值倍数，越大越不敏感。
- `min_abs_deviation`：最小绝对偏差。事件量通常较大，`events_per_hour` 这类小数指标应设置得很小。
- `trend_window`：趋势预测回看窗口。
- `history_window`：历史异常清洗窗口。

## 算法说明

`volume_spike` 和 `events_per_hour` 当前使用周期相位趋势检测：

1. 预测某一天时，只使用该日之前的历史数据，不使用未来数据。
2. 如果设置 `spike_period=7`，则第 N 天只与 N-7、N-14、N-21 等同相位历史比较。
3. 历史数据会先做双向清洗，降低历史尖峰对预测曲线和 safe range 的污染。
4. 单个突变点不会立即改写趋势；只有连续稳定的新值达到 `stable_regime_points` 后，后续预测才切换到新趋势。
5. 绘图中的异常点表示“把该天当作当前判定日、只使用它之前历史数据预测时”得到的异常判断，不是历史清洗阶段的异常标记。

当前工具入口默认使用 `anomaly_detection/core/engine_clean.py`。这个版本把异常判断和趋势切换拆成两个状态：`confirmed` 只保存正常趋势点并用于预测，`candidate` 只保存可能形成新趋势的异常候选点；新趋势确认后只影响后续预测，不会回头取消当前点的异常标记。

## 数据来源

事件数据默认来自 PostgreSQL 视图 `v_clip_wide_api`，数据库配置从：

```text
config/env.json
```

读取。`events_per_hour` 使用 `anomaly_detection/fetchers/events_per_hour.py` 中的数据获取逻辑。

如果数据库不可用，工具会尝试进入 mock 测试逻辑或返回无有效数据提示，具体取决于当前调用模式和本地数据文件是否存在。

## 输出文件

运行后会在 `agent_workspace/` 下生成：

- `last_day_anomaly_summary_{fleet_id}_{timestamp}.json`：单次工具调用的结构化异常汇总。
- `multi_mode_anomaly_results_{fleet_id}_{timestamp}.json`：`execute.py` 多模式循环后的原始结果 list。
- `scan_data_{mode}_{fleet_id}_{timestamp}.csv`：当 `enable_visualization=True` 时保存的检测底表。
- `{mode}_curve_{fleet_id}_{timestamp}.png`：当 `enable_visualization=True` 时保存的可视化图。

## 在代码中直接调用

```python
from anomaly_detection import FleetAnomalyDetectionTool, build_time_window_params

tool = FleetAnomalyDetectionTool()
date_args = build_time_window_params(90)

result_text = tool._run(
    fleet_id="your_fleet_id",
    start_date=date_args["start_date"],
    end_date=date_args["end_date"],
    detection_mode="volume_spike",
    enable_visualization=False,
    spike_period=7,
    expected_relative_tolerance=0.45,
    threshold_multiplier=8.0,
    min_abs_deviation=30.0,
)
```

后续 LLM 或程序应优先解析 `STRUCTURED_ANOMALY_RESULT_JSON:` 后面的 JSON，而不是依赖自然语言报告。

如果要把多模式检测结果交给 AI/LLM 分析，建议同时提供：

- `agent_workspace/multi_mode_anomaly_results_{fleet_id}_{timestamp}.json`
- `RESULT_SCHEMA.md`

其中 `RESULT_SCHEMA.md` 是通用结构说明文件，只描述返回模板、字段类型、字段含义和可能取值，不依赖任何一次具体运行结果。
