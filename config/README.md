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
| `cognee.yaml` | Cognee |
| `evermemos_native.yaml` | EverMemos Native |
| `evermemos.yaml` | EverMemos |
| `graphiti_local.yaml` | Graphiti Local |
| `graphiti.yaml` | Graphiti |
| `graphrag.yaml` | GraphRAG |
| `hindsight.yaml` | Hindsight |
| `lifemem.yaml` | LifeMem |
| `mem0.yaml` | Mem0 |
| `memos_cloud.yaml` | Memos Cloud |
| `memu_cloud.yaml` | Memu Cloud |
| `mindmemos.yaml` | MindMemos |
| `tencentdb.yaml` | TencentDB Agent Memory |
| `zep_cloud.yaml` | Zep Cloud |



### 配置格式

```yaml
builder: "mem0"                           # Builder 类型
docker_compose: "systems/mem0/server/docker-compose.yaml"  # Docker compose 文件路径
env_template: "systems/mem0/server/.env.example"          # 环境变量模板
docker_wait: 30                           # 等待 Docker 服务就绪的秒数
```

> **注意**：带 `_cloud` 的配置文件包含敏感 API Key，已加入 `.gitignore`。

## datasets/ — 数据集元数据

定义各数据集的元信息，如样本配置、评估类型等。具体数据样式可查看根目录下datasets文件夹