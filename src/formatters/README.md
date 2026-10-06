# formatters/ — 检索结果格式化

把检索结果转换成喂给 LLM 的上下文字符串。

| 文件 | 作用 |
|------|------|
| `base.py` | `format_context()`：`SearchResult` 列表 → prompt 上下文文本 |
| `__init__.py` | 统一导出 `format_context` |

说明：ANSWER 阶段调用它拼装上下文；要调整上下文样式（分隔符、字段取舍等），改这里即可。
