# lifebench_event_scource

LifeBench 的「事件粒度」数据源目录。以三人（`孙雨薇` / `于晓薇` / `冯浩然`）2025 全年的生活数据为对象，提供从「按天汇总」到「按事件」两种粒度的会话数据。

## 目录结构

```
lifebench_event_scource/
├── files/
│   ├── daily_event_{pinyin}.json        # 源事件数据（最细粒度，每人一份）
│   ├── lifebench_raw_{pinyin}.json      # 原始样本：按天组织（每人一份）
│   ├── build_raw_event_persons.py       # 从 3380QA 抽取各人原始样本
│   └── build_lifebench_event.py         # 转换脚本：由 raw + daily_event 生成 lifebench_event.json
├── lifebench_event.json                 # 事件粒度样本（生成产物，3 个 sample）
└── README.md
```

## 人物构成

| 人物 | daily_event 事件数 | raw QA | raw 条数 |
|---|---:|---:|---:|
| 孙雨薇 | 5716 | 328 | 7466 |
| 于晓薇 | 5931 | 332 | 7097 |
| 冯浩然 | 5722 | 348 | 7688 |

---

## 文件说明

### files/daily_event_{pinyin}.json —— 源事件数据（每人一份）

逐事件记录某人 2025 年的生活，覆盖 `2025-01-01` ~ `2025-12-31`。

每条事件的字段（三人格式一致）：

| 字段 | 说明 |
| --- | --- |
| `event_id` | 全局编号（按日期顺序递增，每人从 1 起） |
| `name` | 事件名称 |
| `date` | 时间区间列表，如 `"2025-01-01 08:15:00至2025-01-01 09:25:00"`；个别事件含多个时段 |
| `type` | 事件类别 |
| `description` | 事件描述 |
| `participant` | 参与者列表 `[{ "name", "relation" }]` |
| `location` | 地点 |
| `atomic_id` | 原子事实编号列表（部分事件为空） |

示例（`孙雨薇`）：

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

### files/lifebench_raw_{pinyin}.json —— 原始样本（每人一份）

每人 1 个样本（`sample_id` 为中文名），会话按天拆分，结构与 `lifebench_dense` 的 raw 一致。

- `conversation`：`speaker_a` / `speaker_b`，以及 365 个 `session_N_date_time` + `session_N` 键值对。
- 每个 `session_N` 是当天的全部记录，条目为 `{speaker, dia_id, text}`。

单条记录示例：

```json
{
  "speaker": "孙雨薇",
  "dia_id": "2025-01-01_fitness_health0",
  "text": "孙雨薇在2025-01-01的健康数据总结为：……"
}
```

### lifebench_event.json —— 事件粒度样本（生成产物）

由 `daily_event_*.json` + `lifebench_raw_*.json` 生成，**3 个 sample**，共 **17369** 条事件记录（5716 + 5931 + 5722）。

每条记录：

```json
{
  "speaker": "孙雨薇",
  "dia_id": "2025-01-01_event1",
  "text": "孙雨薇在2025-01-01 08:15:00至2025-01-01 09:25:00的活动记录：在晨光中醒来，……"
}
```

- `dia_id`：`{日期}_event{编号}`，编号按天从 1 重新计数。
- `text`：`{人名}在{时间区间}的活动记录：{description}`，时间区间具体到秒；含多个时段的事件用 `、` 连接。
- `qa` 原样保留自对应 raw，三人合计 **1008** 条。
- 空 session（当日无事件）共 4 个：孙雨薇 `2025-10-06`、于晓薇 `2025-11-08`、冯浩然 `2025-08-05` / `2025-12-25`。

## QA 字段

三个样本的 `qa` 均为列表（328 / 332 / 348 条），单条结构：

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

> 注意：`lifebench_event.json` 的 `qa.evidence` 仍引用 raw 中旧的 `dia_id`（如 `agent_chat58`），这些编号在事件粒度样本中已不存在，尚未对齐。

## 数据关系

```
daily_event_{pinyin}.json ──(生成 conversation 的 session 内容)──▶ lifebench_event.json
lifebench_raw_{pinyin}.json ──(提供样本骨架：speaker / session 日期 / qa)──▶ lifebench_event.json
```

## 转换脚本

- `build_raw_event_persons.py` — 从 `lifebench_locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json` 抽取各人原始样本，生成 `files/lifebench_raw_{pinyin}.json`。
- `build_lifebench_event.py` — 由 per-person 的 `daily_event_{pinyin}.json` + `lifebench_raw_{pinyin}.json` 生成 `lifebench_event.json`。默认构建三人（`PERSONS` 字典）；`-e` / `-r` 可切换到单文件模式。

```bash
# 抽取原始样本（默认三人）
python files/build_raw_event_persons.py

# 生成事件粒度样本（默认三人）
python files/build_lifebench_event.py

# 仅某一人
python files/build_lifebench_event.py --sample-id 孙雨薇
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-e, --events` | 单文件模式：源事件数据路径 | 无（默认按 `PERSONS` 逐人取 `daily_event_{pinyin}.json`） |
| `-r, --raw` | 单文件模式：原始样本路径 | 无（默认按 `PERSONS` 逐人取 `lifebench_raw_{pinyin}.json`） |
| `-o, --output` | 输出路径 | `../lifebench_event.json` |
| `--speaker` | 覆盖 speaker 名称（仅单文件模式） | 取样本的 `sample_id` |
| `--sample-id` | 仅转换指定样本 | 转换 `PERSONS` 全部 |
| `--indent` | JSON 缩进空格数 | `2` |

脚本内置转换规则（与上文 `lifebench_event.json` 一致）：

1. 保留原始样本骨架（`sample_id` / `speaker_a` / `speaker_b` / `qa` / 各 session 日期）。
2. 每个 `session_N` 的内容替换为当天的「事件记录」，事件按 `event_id` 排序；同一事件跨多个时段（同天）只计一次。
3. `dia_id = "{日期}_event{编号}"`，编号按天从 1 重新计数。
4. `text = "{人名}在{时间区间}的活动记录：{description}"`，时间区间具体到秒，多时段用 `、` 连接。
5. 某天无事件时，对应 session 为空列表。