# builders/ — 被测系统运行环境构建器

评测开始前把被测系统准备好（起 Docker、生成 `.env`、注入环境变量、等服务就绪），评测结束时负责清理。

| 文件 | 作用 |
|------|------|
| `base_builder.py` | `BaseBuilder` 抽象接口（`build` / `cleanup` / `get_status`） |
| `registry.py` | 构建器注册表（懒加载），`create_builder(name, config, project_root)` |
| `{system}_builder.py` | 各系统的环境准备逻辑，用 `@register_builder("{system}")` 注册 |

约定：

- 注册名对应 `config/systems/{system}.yaml` 的 `builder:` 字段；没有该字段时 cli 会跳过环境准备，直接进流水线
- 各系统形态差异较大：有的起 docker-compose + 服务进程并轮询就绪，有的只在进程内校验依赖 / venv（如需要在特定虚拟环境里跑）
- Builder 还能往被测系统注入配置，例如把系统配置里的 `llm_proxy_url` 写进容器或服务的环境变量
