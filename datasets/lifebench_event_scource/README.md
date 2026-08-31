# lifebench_event_scource

LifeBench 的「事件粒度」数据源目录。以单人（`孙雨薇`）2025 全年的生活数据为对象，提供从「按天汇总」到「按事件」两种粒度的会话数据。

## 目录结构

```
lifebench_event_scource/
├── files/
│   ├── daily_event.json           # 源事件数据（最细粒度，5716 条事件）
│   ├── lifebench_raw.json         # 原始样本：按天组织（8 类记录，共 7466 条）
│   └── build_lifebench_event.py   # 转换脚本：由上面两个文件生成 lifebench_event.json
├── lifebench_event.json           # 事件粒度样本（生成产物）
└── README.md
```

## 文件说明

### files/daily_event.json —— 源事件数据

逐事件记录 `孙雨薇` 2025 年的生活，共 **5716** 条事件，覆盖 `2025-01-01` ~ `2025-12-31`（364 天，缺失 `2025-10-06`）。

每条事件的字段：

| 字段 | 说明 |
| --- | --- |
| `event_id` | 全局编号（1–5716，按日期顺序递增） |
| `name` | 事件名称 |
| `date` | 时间区间列表，如 `"2025-01-01 08:15:00至2025-01-01 09:25:00"`；个别事件含多个时段 |
| `type` | 事件类别 |
| `description` | 事件描述（第二人称叙述） |
| `participant` | 参与者列表 `[{ "name", "relation" }]` |
| `location` | 地点 |
| `atomic_id` | 原子事实编号列表（部分事件为空） |

示例：

```json
{
  "event_id": "1",
  "name": "清晨起床、早餐与家庭互动",
  "date": ["2025-01-01 08:15:00至2025-01-01 09:25:00"],
  "type": "Family&Living Situation",
  "description": "在晨光中醒来，感受到假日慵懒，拉开窗帘让阳光洒满房间。……",
  "participant": [
    { "name": "孙雨薇", "relation": "自己" },
    { "name": "孙明远", "relation": "父亲" },
    { "name": "李秀兰", "relation": "母亲" }
  ],
  "location": "云南省昆明市五华区华山西路7号翠湖社区家中",
  "atomic_id": []
}
```

事件类别分布（`Family` 一类存在拼写变体，已合并统计）：

| 类别 | 数量 |
| --- | --- |
| Personal Life | 2292 |
| Family & Living Situation | 1228 |
| Health | 780 |
| Career | 580 |
| Education | 406 |
| Relationships | 304 |
| Finance | 58 |
| Other | 57 |
| Unexpected Events | 11 |

### lifebench_raw.json —— 原始样本（按天组织）

1 个样本（`sample_id: "孙雨薇"`），会话按天拆分。

- `conversation`：`speaker_a` / `speaker_b`，以及 365 个 `session_N_date_time` + `session_N` 键值对。
- 每个 `session_N` 是当天的全部记录，共 **7466** 条，8 类：

| 记录类型（dia_id 后缀） | 数量 | 含义 |
| --- | --- | --- |
| `agent_chat` | 3360 | 与助手对话 |
| `note` | 928 | 笔记 |
| `calendar` | 694 | 日程 |
| `photo` | 690 | 照片 |
| `push` | 560 | 推送 |
| `sms` | 541 | 短信 |
| `fitness_health` | 364 | 健康数据（每天 1 条） |
| `call` | 329 | 通话 |

单条记录示例：

```json
{
  "speaker": "孙雨薇",
  "dia_id": "2025-01-01_fitness_health0",
  "text": "孙雨薇在2025-01-01的健康数据总结为：……"
}
```

### lifebench_event.json —— 事件粒度样本（生成）

由 `files/daily_event.json` 生成，结构沿用 `lifebench_raw.json` 的骨架（`speaker_a` / `speaker_b` / 365 个 session / `qa`），但每个 `session_N` 的内容由「按天日记」替换为「事件记录」。

每条记录：

```json
{
  "speaker": "孙雨薇",
  "dia_id": "2025-01-01_event1",
  "text": "孙雨薇在2025-01-01 08:15:00至2025-01-01 09:25:00的活动记录：在晨光中醒来，……"
}
```

- `dia_id`：`{日期}_event{编号}`，编号按天从 1 重新计数。
- `text`：`孙雨薇在{时间区间}的活动记录：{description}`，时间区间具体到秒；含多个时段的事件用 `、` 连接。
- 共 **5716** 条，与 `daily_event.json` 事件数一致；`2025-10-06` 无事件，对应 session 为空列表。
- `qa`（328 条）原样保留自 `lifebench_raw.json`。

## QA 字段

两个样本文件均包含 `qa` 列表（328 条），单条结构：

| 字段 | 说明 |
| --- | --- |
| `question` | 问题（含提问时间前缀） |
| `answer` | 标准答案 |
| `evidence` | 引用的 `dia_id` 列表 |
| `category` | 问题类别编码（多标签，整数 0–8） |
| `question_type` | 问题类型（多标签），如 `Single_hop`、`Multi_hop`、`Temporal`、`Conflict`、`Causal`、`Pattern_recognition(Non-declarative)`、`Knowledge_update`、`Unanswerable` |
| `question_id` | 问题唯一 ID |
| `ask_time` | 提问时间 |
| `score_points` | 评分点列表 `[{ "description", "score" }]` |

> 注意：`lifebench_event.json` 的 `qa.evidence` 仍引用 `lifebench_raw.json` 中旧的 `dia_id`（如 `agent_chat58`），这些编号在事件粒度样本中已不存在，尚未对齐。

## 数据关系

```
daily_event.json ──(生成 conversation 的 session 内容)──▶ lifebench_event.json
lifebench_raw.json ──(提供样本骨架：speaker / session 日期 / qa)──▶ lifebench_event.json
```

## 转换脚本

`files/build_lifebench_event.py` 用于从 `daily_event.json` + `lifebench_raw.json` 生成 `lifebench_event.json`。默认路径基于脚本自身位置解析，任意目录下运行均可。

```bash
# 使用默认路径（files/daily_event.json + files/lifebench_raw.json → ../lifebench_event.json）
python files/build_lifebench_event.py

# 自定义输入/输出路径
python files/build_lifebench_event.py \
  -e files/daily_event.json \
  -r files/lifebench_raw.json \
  -o lifebench_event.json

# 覆盖 speaker 或仅转换指定样本
python files/build_lifebench_event.py --speaker 孙雨薇 --sample-id 孙雨薇
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-e, --events` | 源事件数据路径 | `files/daily_event.json` |
| `-r, --raw` | 原始样本路径 | `files/lifebench_raw.json` |
| `-o, --output` | 输出路径 | `../lifebench_event.json` |
| `--speaker` | 覆盖 speaker 名称 | 取样本的 `sample_id` |
| `--sample-id` | 仅转换指定样本 | 转换全部 |
| `--indent` | JSON 缩进空格数 | `2` |

脚本内置转换规则（与上文 `lifebench_event.json` 一致）：

1. 保留原始样本骨架（`sample_id` / `speaker_a` / `speaker_b` / `qa` / 各 session 日期）。
2. 每个 `session_N` 的内容替换为当天的「事件记录」，事件按 `event_id` 排序；同一事件跨多个时段（同天）只计一次。
3. `dia_id = "{日期}_event{编号}"`，编号按天从 1 重新计数。
4. `text = "{speaker}在{时间区间}的活动记录：{description}"`，时间区间具体到秒，多时段用 `、` 连接。
5. 某天无事件（如 `2025-10-06`）时，对应 session 为空列表。

运行后会打印摘要（样本数 / 事件记录总数 / 空 session 数）。