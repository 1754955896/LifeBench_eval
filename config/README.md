# Config Directory

Stores configuration files for the evaluation framework and the memory systems under test.

## Directory structure

```
config/
├── systems/          # config files for the systems under test
└── datasets/         # dataset metadata config
```

## systems/ — system-under-test config

Each memory system has a corresponding YAML config file, defining the Builder and Adapter parameters:

| File | System |
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

### Config format

```yaml
builder: "mem0"                           # Builder type
docker_compose: "systems/mem0/server/docker-compose.yaml"  # Docker compose file path
env_template: "systems/mem0/server/.env.example"          # environment variable template
docker_wait: 30                           # seconds to wait for the Docker service to be ready
```

> **Note**: config files with `_cloud` or `_blocking` contain sensitive API keys and have been added to `.gitignore`.

## datasets/ — dataset metadata

Defines each dataset's metadata, such as sample config, evaluation type, etc.
