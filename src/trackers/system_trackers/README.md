# system_trackers/ — 系统级资源追踪器

采集被测系统特有的资源指标（容器负载、数据库 / 索引体积、LLM token 等），供 GlobalMonitor 汇总。

| 文件 | 作用 |
|------|------|
| `base.py` | `SystemTracker` 抽象接口、`@register_tracker` 注册表、`SystemSnapshot` / `OpRecord` 数据结构 |
| `registry.py` | `get_tracker(name, config)` / `list_trackers()` |
| `utils/llm_proxy.py` | 客户端工具：向 LLM 代理的 `/token-stats` 查询累计 token 用量 |
| `{system}.py` | 各系统的 tracker 实现，默认指标见 `default.py`（进程 CPU / 内存 + LLM token） |

约定：

- 注册名与 `config/systems/{system}.yaml` 的系统名一致；未注册的系统由 cli 回退到 `default`
- 新增 tracker：继承 `DefaultTracker` + 加 `@register_tracker("{system}")`，并在本目录 `__init__.py` 里 import（只加装饰器不导入不会注册）
- 需要修正 LLM 请求格式的系统，可在 tracker 里实现 `transform_llm_request` / `wrap_llm_response`，由 `python -m src.utils.llm_proxy -s {system}` 加载
- 各系统的实现文件会随被测系统增删，此处不逐一列举
