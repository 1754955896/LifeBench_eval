# 配置目录

存放评测框架和被测记忆系统的配置文件。

## 目录结构

```
config/
├── systems/          # 被测系统的配置文件
└── datasets/         # 数据集元数据配置
```

## systems/ — 被测系统配置

每个记忆系统对应一个 YAML 配置文件，定义 Builder 和 Adapter 的参数：

| 文件 | 系统 |
|------|------|
| `mem0.yaml` | Mem0 |
| `cognee.yaml` | Cognee |
| `graphiti.yaml` | Graphiti |
| `graphrag.yaml` | GraphRAG |
| `hindsight.yaml` | Hindsight |
| `evermemos.yaml` | EverMemos |
| `evermemos_native.yaml` | EverMemos Native |
| `mindmemos.yaml` | MindMemos |
| `tencentdb.yaml` | TencentDB Agent Memory |
| `memos_cloud.yaml` | Memos Cloud |
| `memos_cloud_blocking.yaml` | Memos Cloud (Blocking) |
| `zep_cloud.yaml` | Zep Cloud |
| `lifemem.yaml` | LifeMem |

### 配置格式

```yaml
builder: "mem0"                           # Builder 类型
docker_compose: "systems/mem0/server/docker-compose.yaml"  # Docker compose 文件路径
env_template: "systems/mem0/server/.env.example"          # 环境变量模板
docker_wait: 30                           # 等待 Docker 服务就绪的秒数
```

> **注意**：带 `_cloud` 或 `_blocking` 的配置文件包含敏感 API Key，已加入 `.gitignore`。

## datasets/ — 数据集元数据

定义各数据集的元信息，如样本配置、评估类型等。