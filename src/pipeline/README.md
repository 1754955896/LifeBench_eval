# pipeline/ — 评测流水线

| 文件 | 作用 |
|------|------|
| `stages.py` | `Stage` 枚举：ADD / SEARCH / ANSWER / EVALUATE，含 `from_string()` 与前后序查询 |
| `config.py` | `PipelineConfig`：阶段列表、断点开关、run_name、类别过滤、smoke / 对话范围等运行时配置 |
| `checkpoint.py` | `CheckpointManager`：记录已完成进度（add/search 到日期级、answer 到 QA 级），支持中断后续跑，落 `checkpoint_{run_name}.json` |
| `runner.py` | 主执行器 `Pipeline`：按日期交叉执行 ADD + SEARCH，再 ANSWER、EVALUATE，结果增量落盘 |
| `runner_multi_thread.py` | 另一套多线程并发实现（`PipelineMultiThread`），根目录 cli 未使用 |
| `__init__.py` | 导出 `Pipeline` / `PipelineConfig` / `Stage` / `CheckpointManager` |

流程要点：

- ADD 与 SEARCH 按日期推进：先写入当日 session，再检索当日问题，检索结果增量追加到磁盘
- 只跑部分阶段时（`--stages`），answer / evaluate 会从输出目录读取上一阶段的结果文件，不重跑前置阶段
- 重跑同一输出目录会自动跳过已完成部分（checkpoint + 已有结果文件）
