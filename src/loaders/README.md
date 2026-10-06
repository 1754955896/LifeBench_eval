# loaders/ — 数据集加载器

把原始数据集文件解析成框架统一的数据结构。

| 文件 | 作用 |
|------|------|
| `base.py` | `BaseLoader` 抽象接口：`load(path, name, dataset_format, ...) -> Dataset` |
| `registry.py` | 注册表 + `load_dataset()` 入口：按 `config/datasets/*.yaml` 的 `data.format` 选择加载器 |
| `locomo.py` | LoCoMo 格式解析；LifeBench 数据集也走这里（`format: lifebench` 与 `locomo` 都映射到该加载器） |

约定：

- `load_dataset()` 返回统一的 `Dataset`（samples + qa_pairs），后续 pipeline 不感知原始格式
- 支持新数据格式时，加 `{format}_loader.py` 并登记到 `registry.py` 的 `_LOADER_MODULES` 与格式映射
