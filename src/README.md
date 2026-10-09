# LifeBench_eval Core Source

The core code of the evaluation framework, implementing automated evaluation of memory systems.

## Directory structure

```
src/
├── adapters/      # adapters: connect the framework to each memory system
├── builders/      # builders: start and manage the runtime environment of systems under test
├── pipeline/      # pipeline: ADD → SEARCH → ANSWER → EVALUATE four-stage scheduling
├── evaluators/    # evaluators: judge answer quality
├── loaders/       # data loaders: load evaluation datasets
├── formatters/    # formatters: format retrieval results into context
├── models/        # data models: define the framework's internal data structures
└── utils/         # utility functions: config, logging, retry, etc.
```

## Core interfaces

### Adapter

Defines the interface between the framework and memory systems; each system under test must implement the following methods:

```python
class BaseAdapter:
    async def add_chunks(self, chunks: List[MessageChunk]) -> None:
        """Ingest message chunks into the memory system"""

    async def search(self, query: str, user_id: str, top_k: int = 5) -> List[SearchResult]:
        """Search memories"""

    async def answer(self, query: str, context: List[str]) -> str:
        """Generate an answer based on context"""

    async def cleanup(self) -> None:
        """Clean up resources"""
```

### Builder

Starts the runtime environment of the memory system under test (usually Docker):

```python
class BaseBuilder:
    async def build(self) -> bool:
        """Start the system, returns whether it succeeded"""

    async def cleanup(self) -> None:
        """Clean up system resources"""
```

### Pipeline

Four-stage evaluation flow, details in `pipeline/runner.py`:

1. **ADD + SEARCH**: interleaves data ingestion and retrieval by date
2. **ANSWER**: generates answers based on the retrieval results
3. **EVALUATE**: uses an LLM Judge to evaluate answer quality
