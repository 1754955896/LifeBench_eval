# adapters/ — 记忆系统适配器

把统一评测接口翻译成各记忆系统自己的调用方式（HTTP API、SDK 或进程内直调）。

| 文件 | 作用 |
|------|------|
| `base.py` | `BaseAdapter` 抽象接口（`add_chunks` / `search` / `answer` / `close`）及 `ChunkedMessage` 等公共数据结构 |
| `registry.py` | 适配器注册表（懒加载）：名称 → 模块路径映射，`create_adapter(name, config)` 按需导入并实例化 |
| `example_adapter.py` | 新系统接入的模板，复制改造成 `{system}_adapter.py` |
| `{system}_adapter.py` | 各系统实现，用 `@register_adapter("{system}")` 注册 |

约定：

- 注册名必须与 `config/systems/{system}.yaml` 的 `adapter:` 字段一致
- 新增一个系统：写 `{system}_adapter.py` → 登记到 `registry.py` 的 `_ADAPTER_MODULES` → 补系统配置
