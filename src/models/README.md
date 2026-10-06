# models/ — 核心数据模型

框架内部流通的数据结构，全部是 dataclass。

| 文件 | 作用 |
|------|------|
| `message.py` | `Message` / `Conversation`：消息与会话 |
| `dataset.py` | `Dataset` / `QAPair`：数据集与问答对（含题目类型、证据等元信息） |
| `search.py` | `SearchResult` / `RetrievedMemory`：检索结果与单条记忆 |
| `answer.py` | `AnswerResult`：回答结果（`answer` / `golden_answer` / 元信息） |
| `evaluation.py` | `EvaluationResult` / `QuestionTypeStats`：判分结果与分题目类型统计 |
| `__init__.py` | 统一导出 |

说明：这些模型落盘后就是输出目录里的 `search_results.json` / `answer_results.json` / `eval_results.json`（字段说明见 `results_clean/README.md`）。
