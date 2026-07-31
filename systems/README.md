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

## 各系统配置文件

不同系统有不同的配置方式：

### MindMemOS

配置文件：`systems/MindMemOS/config/mindmemos/dev.yaml`

运行前需配置此文件，参考 `dev.yaml.example`：
```bash
cp systems/MindMemOS/config/mindmemos/dev.yaml.example systems/MindMemOS/config/mindmemos/dev.yaml
# 编辑 dev.yaml 填入 API Key 等配置
```

**手动启动（如 builder 启动失败，可手动启动服务）：**

启动 Docker 服务（Qdrant、Neo4j、Kafka）：
```bash
docker compose --env-file systems/MindMemOS/.env -f systems/MindMemOS/dockers/docker-compose.memory.yml up -d --wait qdrant neo4j kafka kafka-ui kafka-exporter
```

启动 API Server：
```bash
cd systems/MindMemOS ; .venv/Scripts/python.exe -m uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### Hindsight

配置文件：`systems/hindsight/.env`

运行前需配置此文件，参考 `.env.example`：
```bash
cp systems/hindsight/.env.example systems/hindsight/.env
# 编辑 .env 填入 API Key 等配置
```

Hindsight 使用本地模式（pg0 嵌入式数据库），无需 Docker。

