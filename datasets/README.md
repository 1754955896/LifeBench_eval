# 数据集目录

存放评测用的数据集文件。

## 目录结构

```
datasets/
├── lifebench_locomo_format/       # LifeBench LoCoMo 标准格式数据集
├── lifebench_locomo_3people/      # 3人对话版本的 LifeBench LoCoMo
├── locomo/                        # 原始 LoCoMo 数据
└── smoke/                          # 冒烟测试数据集
```

## 数据集说明

### lifebench_locomo_format/

LifeBench 标准格式的 LoCoMo 数据集。

- `lifebench_locomo_conversation_format_v2.0_3380QA.json` — 主要数据集（约 27MB）

### lifebench_locomo_3people/

3人对话版本的 LifeBench LoCoMo 数据集。

- `lifebench_locomo_3people.json`

### locomo/

原始 LoCoMo（Long-term Conversation Model）数据集。

- `locomo10.json`
- `first_sample.json`

### smoke/

用于冒烟测试的小规模数据集，用于快速验证框架功能。

- `smoke_data.json`
- `locomo_smoke.json`

## 数据格式

数据集采用统一的 JSON 格式，包含：

- **sessions**：对话会话列表，每个会话包含日期和消息列表
- **qa_pairs**：问答对，包含问题、期望答案和答案类型标签
- **samples**：样本列表，每个样本关联特定用户/场景