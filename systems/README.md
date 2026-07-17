# 被测系统目录

存放各被测记忆系统的源码和部署配置。

## 目录结构

```
systems/
├── mem0/                   # Mem0 记忆系统
├── cognee/               # Cognee 记忆系统
├── graphiti/             # Graphiti 记忆系统
├── graphrag/             # GraphRAG 记忆系统
├── hindsight/           # Hindsight 记忆系统
├── MemOS/               # MemOS 记忆系统
├── memU/                # MemU 记忆系统
├── memU-server/         # MemU Server 版本
├── memory/              # Memory 记忆系统
├── MindMemOS/          # MindMemOS 记忆系统
├── supermemory/         # Supermemory 记忆系统
├── TencentDB-Agent-Memory/  # TencentDB Agent Memory
└── EverMemOS_bz/        # EverMemOS (bz 版本)
```

## 系统说明

每个子系统通常包含：

- **Docker 部署配置**：`docker-compose.yaml` 或 `Dockerfile`
- **服务端代码**：记忆系统的 API 服务
- **客户端 SDK**：如需额外的客户端库
- **Helm Chart**（可选）：Kubernetes 部署配置

## 与框架的集成

Builder 根据 `config/systems/{system}.yaml` 中的配置启动对应的系统：

```yaml
builder: "mem0"
docker_compose: "systems/mem0/server/docker-compose.yaml"
env_template: "systems/mem0/server/.env.example"
docker_wait: 30
```

框架通过环境变量将 API Key 等配置注入到 Docker 容器中。

## 注意事项

- `systems/` 目录包含各个记忆系统的完整代码
- 带 `_cloud` 的配置（`config/systems/*_cloud*.yaml`）包含敏感 API Key，不会提交到版本控制