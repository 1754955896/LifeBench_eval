# LifeBench_eval 核心源码

评估框架的核心代码，实现对记忆系统的自动化评测。

## 目录结构

```
src/
├── adapters/      # 适配器：连接框架与各记忆系统
├── builders/      # 构建器：启动和管理被测系统的运行环境
├── pipeline/      # 流水线：ADD → SEARCH → ANSWER → EVALUATE 四阶段调度
├── evaluators/    # 评估器：评判答案质量
├── loaders/       # 数据加载器：加载评测数据集
├── formatters/    # 格式化器：格式化检索结果为上下文
├── models/        # 数据模型：定义框架内部数据结构
└── utils/         # 工具函数：配置、日志、重试等
```

## 核心接口

### Adapter（适配器）

定义框架与记忆系统的接口，每个被测系统需实现以下方法：

```python
class BaseAdapter:
    async def add_chunks(self, chunks: List[MessageChunk]) -> None:
        """摄入消息片段到记忆系统"""

    async def search(self, query: str, user_id: str, top_k: int = 5) -> List[SearchResult]:
        """搜索记忆"""

    async def answer(self, query: str, context: List[str]) -> str:
        """基于上下文生成回答"""

    async def cleanup(self) -> None:
        """清理资源"""
```

### Builder（构建器）

启动被测记忆系统的运行环境（通常是 Docker）：

```python
class BaseBuilder:
    async def build(self) -> bool:
        """启动系统，返回是否成功"""

    async def cleanup(self) -> None:
        """清理系统资源"""
```

### Pipeline（流水线）

四阶段评测流程，详见 `pipeline/runner.py`：

1. **ADD + SEARCH**：按日期交叉执行数据摄入与检索
2. **ANSWER**：基于检索结果生成答案
3. **EVALUATE**：使用 LLM Judge 评估答案质量