# trackers/ — 资源追踪

`cli.py --enable-tracker` 时启用，数据落在 `{output_dir}/tracker/`。

| 文件 | 作用 |
|------|------|
| `per_op_tracker.py` | `PerOpTracker`：记录每次 add/search 的耗时与元数据 → `tracker/tracker_records.json`；同时在全局时间线插入 op_start / op_end 标记 |
| `global_monitor.py` | `GlobalMonitor`：后台线程周期采样 CPU / 内存 / 存储 → `tracker/global_resource_timeline.json`，结束时汇总 summary |
| `system_trackers/` | 系统级 tracker：接口、注册表与各系统实现（详见该目录的 README） |
| `__init__.py` | 统一导出 `PerOpTracker` / `GlobalMonitor` / `get_tracker` / `list_trackers` |

说明：

- 采样间隔由 `--tracker-interval` 控制（默认 5s）
- 每次操作的开销看 `tracker_records.json`，资源曲线看 `global_resource_timeline.json`（可按 op 标记切片）
