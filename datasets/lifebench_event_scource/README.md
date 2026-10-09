# lifebench_event_scource

LifeBench's "event-granularity" data-source directory. Built on three people (`孙雨薇` / `于晓薇` / `冯浩然`) across the whole of 2025, it provides conversation data at two granularities: "per-day summary" and "per-event".

## Directory structure

```
lifebench_event_scource/
├── files/
│   ├── daily_event_{pinyin}.json        # source event data (finest granularity, one per person)
│   ├── lifebench_raw_{pinyin}.json      # raw samples: organized by day (one per person)
│   ├── build_raw_event_persons.py       # extract each person's raw sample from 3380QA
│   └── build_lifebench_event.py         # conversion script: from raw + daily_event to lifebench_event.json
├── lifebench_event.json                 # event-granularity samples (generated, 3 samples)
└── README.md
```

## Person composition

| Person | daily_event event count | raw QA | raw entries |
|---|---:|---:|---:|
| 孙雨薇 | 5716 | 328 | 7466 |
| 于晓薇 | 5931 | 332 | 7097 |
| 冯浩然 | 5722 | 348 | 7688 |

---

## File descriptions

### files/daily_event_{pinyin}.json — source event data (one per person)

Records someone's 2025 life event-by-event, covering `2025-01-01` ~ `2025-12-31`.

Fields of each event (identical format across the three people):

| Field | Description |
| --- | --- |
| `event_id` | global number (increments by date order, starting from 1 per person) |
| `name` | event name |
| `date` | time-range list, e.g. `"2025-01-01 08:15:00至2025-01-01 09:25:00"`; some events span multiple periods |
| `type` | event category |
| `description` | event description |
| `participant` | participant list `[{ "name", "relation" }]` |
| `location` | location |
| `atomic_id` | atomic-fact number list (empty for some events) |

Example (`孙雨薇`):

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

### files/lifebench_raw_{pinyin}.json — raw samples (one per person)

One sample per person (`sample_id` is the Chinese name), conversation split by day, structure consistent with `lifebench_dense`'s raw.

- `conversation`: `speaker_a` / `speaker_b`, plus 365 `session_N_date_time` + `session_N` key-value pairs.
- Each `session_N` is the full record for that day, with entries `{speaker, dia_id, text}`.

Single-record example:

```json
{
  "speaker": "孙雨薇",
  "dia_id": "2025-01-01_fitness_health0",
  "text": "孙雨薇在2025-01-01的健康数据总结为：……"
}
```

### lifebench_event.json — event-granularity samples (generated)

Generated from `daily_event_*.json` + `lifebench_raw_*.json`, **3 samples**, **17369** event records in total (5716 + 5931 + 5722).

Each record:

```json
{
  "speaker": "孙雨薇",
  "dia_id": "2025-01-01_event1",
  "text": "孙雨薇在2025-01-01 08:15:00至2025-01-01 09:25:00的活动记录：在晨光中醒来，……"
}
```

- `dia_id`: `{date}_event{number}`, the number restarts from 1 each day.
- `text`: `{person name}在{time range}的活动记录：{description}`, time range precise to the second; multi-period events are joined with `、`.
- `qa` is preserved as-is from the corresponding raw, **1008** entries across the three people.
- 4 empty sessions (no events that day): 孙雨薇 `2025-10-06`, 于晓薇 `2025-11-08`, 冯浩然 `2025-08-05` / `2025-12-25`.

## QA fields

All three samples' `qa` are lists (328 / 332 / 348 entries), with a single-entry structure:

| Field | Description |
| --- | --- |
| `question` | the question (with ask-time prefix) |
| `answer` | the golden answer |
| `evidence` | the cited `dia_id` list |
| `category` | question category encoding (multi-label, integers 0–8) |
| `question_type` | question type (multi-label), e.g. `Single_hop`, `Multi_hop`, `Temporal`, `Conflict`, `Causal`, `Pattern_recognition(Non-declarative)`, `Knowledge_update`, `Unanswerable` |
| `question_id` | question unique ID |
| `ask_time` | ask time |
| `score_points` | scoring-point list `[{ "description", "score" }]` |

> Note: `lifebench_event.json`'s `qa.evidence` still cites the old `dia_id`s in the raw (e.g. `agent_chat58`); these numbers no longer exist in the event-granularity samples and have not been aligned yet.

## Data relationships

```
daily_event_{pinyin}.json ──(generates the conversation's session content)──▶ lifebench_event.json
lifebench_raw_{pinyin}.json ──(provides the sample skeleton: speaker / session dates / qa)──▶ lifebench_event.json
```

## Conversion scripts

- `build_raw_event_persons.py` — extracts each person's raw sample from `lifebench_locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json`, producing `files/lifebench_raw_{pinyin}.json`.
- `build_lifebench_event.py` — generates `lifebench_event.json` from per-person `daily_event_{pinyin}.json` + `lifebench_raw_{pinyin}.json`. Builds the three people by default (`PERSONS` dict); `-e` / `-r` switch to single-file mode.

```bash
# extract raw samples (default: three people)
python files/build_raw_event_persons.py

# generate event-granularity samples (default: three people)
python files/build_lifebench_event.py

# just one person
python files/build_lifebench_event.py --sample-id 孙雨薇
```

| Parameter | Description | Default |
| --- | --- | --- |
| `-e, --events` | single-file mode: source event data path | none (default takes `daily_event_{pinyin}.json` per `PERSONS`) |
| `-r, --raw` | single-file mode: raw sample path | none (default takes `lifebench_raw_{pinyin}.json` per `PERSONS`) |
| `-o, --output` | output path | `../lifebench_event.json` |
| `--speaker` | override the speaker name (single-file mode only) | the sample's `sample_id` |
| `--sample-id` | only convert the specified sample | convert all in `PERSONS` |
| `--indent` | JSON indent spaces | `2` |

Built-in conversion rules (consistent with `lifebench_event.json` above):

1. Keep the raw sample skeleton (`sample_id` / `speaker_a` / `speaker_b` / `qa` / each session date).
2. Each `session_N`'s content is replaced with that day's "event records", events sorted by `event_id`; an event spanning multiple periods (same day) counts only once.
3. `dia_id = "{date}_event{number}"`, number restarts from 1 each day.
4. `text = "{person name}在{time range}的活动记录：{description}"`, time range precise to the second, multi-periods joined with `、`.
5. On a day with no events, the corresponding session is an empty list.
