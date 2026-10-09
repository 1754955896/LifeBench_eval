# Datasets Directory

Stores the dataset files used for evaluation.

## Directory structure

```
datasets/
├── lifebench_locomo_format/       # LifeBench LoCoMo standard-format dataset
├── lifebench_locomo_3people/      # 3-person conversation version of LifeBench LoCoMo
├── lifebench_offline/             # LifeBench offline version (ask_time uniformly set to 2025-12-31)
├── locomo/                        # original LoCoMo data
└── smoke/                          # smoke-test dataset
```

## Dataset descriptions

### lifebench_locomo_format/

LifeBench's standard-format LoCoMo dataset.

- `lifebench_locomo_conversation_format_v2.0_3380QA.json` — primary dataset (~27MB)

### lifebench_locomo_3people/

3-person conversation version of the LifeBench LoCoMo dataset.

- `lifebench_locomo_3people.json`

### lifebench_offline/

LifeBench offline-version dataset (10 people / 3380 QA). Generated from `lifebench_locomo_format`, with every QA's `ask_time` field forced to `2025-12-31`; all other fields unchanged.

- `lifebench_offline.json`

### locomo/

The original LoCoMo (Long-term Conversation Model) dataset.

- `locomo10.json`
- `first_sample.json`

### smoke/

A small-scale dataset for smoke testing, used to quickly validate framework functionality.

- `smoke_data.json`
- `locomo_smoke.json`

## Data format

The datasets use a unified JSON format, containing:

- **sessions**: list of conversation sessions, each containing a date and a message list
- **qa_pairs**: question-answer pairs, containing the question, expected answer, and answer-type label
- **samples**: list of samples, each associated with a specific user/scenario
