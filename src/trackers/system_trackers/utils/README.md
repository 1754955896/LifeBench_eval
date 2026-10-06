# utils/ — 系统追踪器公用工具

| 文件 | 作用 |
|------|------|
| `llm_proxy.py` | `query_llm_proxy()`：GET `{proxy_url}/token-stats`，返回 LLM 代理累计的 token 用量（`prompt_tokens` / `completion_tokens` / `total_tokens` / `request_count`），代理不可达时返回 `None` |
| `__init__.py` | 统一导出 `query_llm_proxy` |

说明：

- 这是 LLM 代理的**客户端**；代理服务本体在 [`src/utils/llm_proxy.py`](../../../utils/llm_proxy.py)
- `default.py` 用它给 DefaultTracker 补 LLM token 指标（snapshot 增量 + `backend_specific_stats`），各系统的 tracker 继承 DefaultTracker 后同样可用
- 代理地址默认 `http://localhost:18443`，也可由系统配置的 `llm_proxy_url` 覆盖