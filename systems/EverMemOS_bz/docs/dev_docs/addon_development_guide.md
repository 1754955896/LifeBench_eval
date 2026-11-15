# Addon Development Guide

## Overview

This guide introduces how to develop addons for MemSys, including environment setup, development workflow, and best practices. Using the Enterprise addon as an example, it demonstrates how to achieve code separation between commercial and open-source features.

## Quick Start (4 Steps)

If you want to quickly start addon development, just follow these 4 steps:

### 1. Clone Both Repositories to the Same Directory

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone <opensource-repo-url> memsys_opensource
git clone <enterprise-repo-url> memsys_enterprise
```

**Important**: Both repositories must be in the same parent directory.

### 2. Create Virtual Environment and Install Dependencies in Opensource

```bash
cd ~/workspace/memsys_opensource
uv sync
```

This will create a virtual environment in the `memsys_opensource/.venv` directory and install all dependencies.

### 3. Editable Install Both Packages to the Same Virtual Environment

**Key**: Both packages must be installed into **the same virtual environment** (because enterprise needs to import opensource modules).

```bash
# First, install opensource
cd ~/workspace/memsys_opensource
source .venv/bin/activate
uv pip install -e .

# Then, install enterprise to the same virtual environment (Important!)
cd ~/workspace/memsys_enterprise
source ../memsys_opensource/.venv/bin/activate
uv pip install -e .
```

**Why do this?**
- Enterprise code needs to `from core.xxx import xxx` to import opensource modules
- Both packages must be in the same Python environment to access each other
- Entry points must be in the same environment to be discovered by the system

### 4. Start the Service

```bash
cd ~/workspace/memsys_opensource
uv run python -m src.run
```

If you see the following logs, the addon loaded successfully:

```
🔌 Starting to load addons entry points...
  ✅ Loaded entrypoint: core
  ✅ Loaded entrypoint: enterprise
✅ Addons entry points loaded, total: 2
```

Now you can start developing! After modifying code in either repository, simply restart the service (no need to reinstall).

---

## Detailed Instructions

Below is a complete explanation of environment setup and development workflow.

## Development Environment Setup

### 1. Clone Code Repositories

First, clone both opensource and enterprise repositories to **the same directory**:

```bash
# Create workspace directory
mkdir -p ~/workspace
cd ~/workspace

# Clone opensource repository
git clone <opensource-repo-url> memsys_opensource

# Clone enterprise repository
git clone <enterprise-repo-url> memsys_enterprise
```

**Important**: Both repositories must be in the same directory to ensure module references resolve correctly during development.

The final directory structure should be:

```
~/workspace/
├── memsys_opensource/
│   ├── src/
│   ├── pyproject.toml
│   └── ...
└── memsys_enterprise/
    ├── src/
    │   └── memsys_enterprise/
    ├── pyproject.toml
    └── ...
```

### 2. Install Opensource Dependencies

Enter the opensource repository and use uv to create a virtual environment and install dependencies:

```bash
cd ~/workspace/memsys_opensource

# Use uv to sync dependencies (will automatically create virtual environment)
uv sync

# Or if virtual environment already exists
uv sync --frozen
```

This will create a virtual environment in the `memsys_opensource/.venv` directory.

### 3. Editable Install Both Packages to the Same Virtual Environment

**Important**: Both opensource and enterprise packages must be installed into **the same virtual environment**.

```bash
# Step 1: In opensource repository root, install opensource
cd ~/workspace/memsys_opensource
uv pip install -e .

# Step 2: In enterprise repository root, also install enterprise to opensource's virtual environment
cd ~/workspace/memsys_enterprise

# Method 1: Use pip directly (recommended)
../memsys_opensource/.venv/bin/pip install -e .

# Method 2: Use uv specifying Python interpreter
uv pip install -e . --python ../memsys_opensource/.venv/bin/python

# Method 3: Activate opensource virtual environment first, then install
source ../memsys_opensource/.venv/bin/activate  # Linux/macOS
pip install -e .
```

**Why must they be in the same virtual environment?**
- Enterprise code needs to `from core.xxx import xxx` to import opensource modules
- If installed in different virtual environments, enterprise cannot find the core module
- Both packages share dependencies, avoiding duplicate installations
- Entry points must be in the same environment to be discovered correctly

**Purpose of editable install**:
- Code changes take effect without reinstallation
- Entry points are registered in the environment
- Can import like normally installed packages

### 4. Verify Installation

Verify both packages are correctly installed in the same virtual environment:

```bash
# Check installed packages in opensource directory
cd ~/workspace/memsys_opensource
uv pip list | grep memsys

# Should see similar output (note both show as editable installs):
# memsys            0.1.0   /path/to/memsys_opensource/src
# memsys-enterprise 0.1.0   /path/to/memsys_enterprise/src/memsys_enterprise
```

Verify entry points are registered:

```bash
# Use opensource Python environment
cd ~/workspace/memsys_opensource

# Method 1: Use uv run
uv run python -c "
from importlib.metadata import entry_points
eps = entry_points(group='memsys.addons')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"

# Method 2: Run after activating virtual environment
source .venv/bin/activate
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='memsys.addons')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"

# Should see output:
# core: src.addon
# enterprise: memsys_enterprise.addon
```

**If you don't see enterprise**: Enterprise is not installed in the correct virtual environment, please repeat step 3.

### 5. Start the Service

Start the service through the opensource repository, which will automatically load the enterprise addon:

```bash
cd ~/workspace/memsys_opensource

# Method 1: Use uv run (recommended)
uv run python -m src.run

# Method 2: Run after activating virtual environment
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
python -m src.run

# Method 3: Use command defined in project.scripts
uv run web
```

After startup, you should see similar output in the logs:

```
🔌 Starting to load addons entry points...
  ✅ Loaded entrypoint: core
  ✅ Loaded entrypoint: enterprise
✅ Addons entry points loaded, total: 2
```

## Addon Development Principles

### Core Concept

Addons are essentially **not** dependencies, but part of the Open Core codebase. The directory structure of addons is completely consistent with the Open Core directory structure - they are **different parts of the same system**.

### Key Mechanisms

1. **Interface Abstraction**: Define interfaces through abstract classes or protocols
2. **Separate Implementation**: Provide different implementations in different repositories
3. **Automatic Replacement**: Through priority mechanism, addon implementations automatically override open core implementations

### Workflow

```
┌─────────────────┐
│ Define Interface│
│   (Open Core)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌─────────────┐
│  Open │  │  Commercial │
│ Source│  │(Enterprise) │
└───────┘  └─────────────┘
    │         │
    └────┬────┘
         │
         ▼
   ┌─────────┐
   │Priority │
   │Mechanism│
   └─────────┘
```

## Steps to Develop an Addon

### Step 1: Interface Abstraction

When you need to differentiate a feature or logic, first perform interface abstraction.

**Define interface in Open Core**:

```python
# memsys_opensource/src/core/interface/repository/memory_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from core.domain.model.memory import Memory

class MemoryRepository(ABC):
    """
    Memory Storage Repository Interface
    Defines CRUD operation specifications for memories
    """
    
    @abstractmethod
    async def save(self, memory: Memory) -> str:
        """
        Save memory
        
        Args:
            memory: Memory object
            
        Returns:
            str: Memory ID
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Find memory by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Optional[Memory]: Memory object, None if not found
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Search memories
        
        Args:
            query: Query text
            limit: Limit on number of results
            
        Returns:
            List[Memory]: List of memories
        """
        pass
```

### Step 2: Open Core Implementation

Provide a basic implementation in Open Core (usually simplified or local version).

```python
# memsys_opensource/src/infra_layer/adapters/out/persistence/repository/local_memory_repository.py
from typing import List, Optional
from core.interface.repository.memory_repository import MemoryRepository
from core.domain.model.memory import Memory
from core.di.component import Component

@Component()
class LocalMemoryRepository(MemoryRepository):
    """
    Local in-memory storage implementation (for development and testing)
    Data is stored in memory and lost after service restart
    """
    
    def __init__(self):
        self._storage = {}  # Simple dictionary storage
    
    async def save(self, memory: Memory) -> str:
        """Save to memory dictionary"""
        memory_id = memory.id or self._generate_id()
        self._storage[memory_id] = memory
        return memory_id
    
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """Find from memory dictionary"""
        return self._storage.get(memory_id)
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Simple full-text matching search"""
        results = []
        for memory in self._storage.values():
            if query.lower() in memory.content.lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        return results
    
    def _generate_id(self) -> str:
        """Generate simple ID"""
        import uuid
        return str(uuid.uuid4())
```

**Register scan path in Open Core addon**:

```python
# memsys_opensource/src/addon.py
paths_registry.add_scan_path(
    os.path.join(get_base_scan_path(), "infra_layer/adapters/out/persistence")
)
```

### Step 3: Enterprise Implementation

Provide commercial-grade implementation in Enterprise (usually distributed, cloud-native version).

```python
# memsys_enterprise/src/memsys_enterprise/infra_layer/adapters/out/persistence/repository/cloud_memory_repository.py
from typing import List, Optional
from core.interface.repository.memory_repository import MemoryRepository
from core.domain.model.memory import Memory
from core.di.component import Component

@Component()
class CloudMemoryRepository(MemoryRepository):
    """
    Cloud-based distributed storage implementation
    Uses MongoDB + Elasticsearch + Milvus for high-availability storage and search
    """
    
    def __init__(
        self,
        mongo_client,      # Inject MongoDB client
        es_client,         # Inject Elasticsearch client
        milvus_client,     # Inject Milvus client
    ):
        self.mongo = mongo_client
        self.es = es_client
        self.milvus = milvus_client
    
    async def save(self, memory: Memory) -> str:
        """Save to distributed storage"""
        # 1. Save to MongoDB (primary storage)
        memory_id = await self._save_to_mongo(memory)
        
        # 2. Index to Elasticsearch (full-text search)
        await self._index_to_elasticsearch(memory_id, memory)
        
        # 3. Save vectors to Milvus (vector search)
        await self._save_to_milvus(memory_id, memory)
        
        return memory_id
    
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """Query from MongoDB"""
        return await self._find_from_mongo(memory_id)
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Hybrid search: vector search + full-text search + reranking"""
        # 1. Vector search (semantic similarity)
        vector_results = await self._vector_search(query, limit * 2)
        
        # 2. Full-text search (keyword matching)
        text_results = await self._text_search(query, limit * 2)
        
        # 3. Hybrid reranking
        final_results = self._rerank(vector_results, text_results, limit)
        
        return final_results
    
    # ... other private method implementations ...
```

**Register scan path in Enterprise addon**:

```python
# memsys_enterprise/src/memsys_enterprise/addon.py
di_registry.add_scan_path(
    os.path.join(enterprise_base_path, "infra_layer/adapters/out/persistence")
)
```

### Step 4: Priority Mechanism

When both repositories provide implementations for the same interface, the later-loaded addon (Enterprise) automatically overrides the earlier one (Core).

**Implementation principle**:
1. The DI container replaces implementations when encountering the same interface during component scanning
2. Enterprise addon loads after Core addon
3. `CloudMemoryRepository` replaces `LocalMemoryRepository`

**No need to care about specific implementation when using**:

```python
# Business layer code (same in both Open Core and Enterprise)
from core.interface.repository.memory_repository import MemoryRepository
from core.di.injector import inject

class MemoryService:
    def __init__(self):
        # Auto-inject, runtime decides which implementation to use
        self.repository: MemoryRepository = inject(MemoryRepository)
    
    async def save_memory(self, content: str) -> str:
        memory = Memory(content=content)
        # Development environment: uses LocalMemoryRepository
        # Production environment: uses CloudMemoryRepository
        return await self.repository.save(memory)
```

## Directory Structure Standards

### Open Core Structure

```
memsys_opensource/
├── src/
│   ├── addon.py                      # Core addon registration
│   ├── core/                         # Core domain layer
│   │   ├── interface/                # Interface definitions (key)
│   │   │   ├── repository/           # Repository interfaces
│   │   │   ├── service/              # Service interfaces
│   │   │   └── controller/           # Controller interfaces
│   │   ├── domain/                   # Domain models
│   │   ├── di/                       # Dependency injection
│   │   ├── addons/                   # Addon mechanism
│   │   └── ...
│   ├── infra_layer/                  # Infrastructure layer
│   │   └── adapters/
│   │       ├── input/                # Input adapters
│   │       └── out/                  # Output adapters
│   │           └── persistence/      # Persistence implementations
│   ├── agentic_layer/                # Agent layer
│   ├── biz_layer/                    # Business layer
│   └── component/                    # Common components
└── pyproject.toml
```

### Enterprise Structure (Mirrors Open Core)

```
memsys_enterprise/
├── src/
│   └── memsys_enterprise/
│       ├── addon.py                  # Enterprise addon registration
│       └── infra_layer/              # Infrastructure layer (corresponds to Open Core)
│           └── adapters/
│               ├── input/            # Input adapters (commercial implementation)
│               │   ├── api/          # Additional APIs
│               │   └── mcp/          # Additional protocols
│               └── out/              # Output adapters (commercial implementation)
│                   ├── persistence/  # Distributed persistence
│                   └── search/       # Advanced search
└── pyproject.toml
```

**Key Principles**:
- Enterprise directory structure **mirrors** Open Core
- Only includes parts that need to be replaced or added
- Maintains consistent hierarchy for easy understanding and maintenance

## Configuring Entry Points

### Open Core Configuration

```toml
# memsys_opensource/pyproject.toml
[project]
name = "memsys"
version = "0.1.0"
# ... other configurations ...

[project.entry-points."memsys.addons"]
core = "src.addon"
```

### Enterprise Configuration

```toml
# memsys_enterprise/pyproject.toml
[project]
name = "memsys-enterprise"
version = "0.1.0"
# ... other configurations ...

[project.entry-points."memsys.addons"]
enterprise = "memsys_enterprise.addon"
```

**Note**:
- Entry point group name must be `"memsys.addons"`
- Entry point names (like `core`, `enterprise`) can be customized
- Entry point values point to modules containing registration code

## Development Workflow

### 1. Daily Development

```bash
# 1. Modify code (Open Core or Enterprise)
vim memsys_opensource/src/infra_layer/...
vim memsys_enterprise/src/memsys_enterprise/infra_layer/...

# 2. Start testing directly (no need to reinstall)
cd memsys_opensource
uv run python -m src.run

# 3. Check logs to confirm addon loading
# Should see "Loaded entrypoint: enterprise"
```

### 2. Adding New Addon Features

```bash
# 1. Define interface in Open Core
vim memsys_opensource/src/core/interface/service/new_service.py

# 2. Provide basic implementation in Open Core
vim memsys_opensource/src/component/new_service_impl.py

# 3. Provide commercial implementation in Enterprise
vim memsys_enterprise/src/memsys_enterprise/component/new_service_impl.py

# 4. Ensure scan paths are configured (if new path needed)
vim memsys_opensource/src/addon.py
vim memsys_enterprise/src/memsys_enterprise/addon.py

# 5. Start testing
cd memsys_opensource
uv run python -m src.run
```

### 3. Debugging Addon Loading

If you find addons are not loading or components are not being replaced:

```python
# Add debug output in code
from core.addons.addons_registry import ADDONS_REGISTRY

# View all loaded addons
all_addons = ADDONS_REGISTRY.get_all()
for addon in all_addons:
    print(f"Addon: {addon.name}")
    if addon.has_di():
        for path in addon.di.get_scan_paths():
            print(f"  DI Path: {path}")
```

Or set log level to DEBUG:

```bash
export LOG_LEVEL=DEBUG
uv run python -m src.run
```

### 4. Load Only Open Core (Without Enterprise)

```bash
# Set environment variable to load only core addon
export MEMSYS_ENTRYPOINTS_FILTER=core

# Start service
cd memsys_opensource
uv run python -m src.run

# Only Open Core implementations will be loaded, not Enterprise
```

## Best Practices

### 1. Interface First

- Think about interface design before developing new features
- Interfaces should be abstract enough, without implementation details
- Interface definitions go in `core/interface/` directory

### 2. Keep Directory Structure Consistent

- Enterprise directory structure should mirror Open Core
- Makes it easy to quickly locate corresponding implementations
- Reduces maintenance costs

### 3. Documentation and Comments

- Interface definitions must have detailed docstrings
- Explain the purpose, parameters, and return values of each method
- Mark which are open-source implementations and which are commercial

### 4. Test Coverage

- Write unit tests for interfaces
- Tests should be valid for both implementations
- Use dependency injection for easy mocking and testing

### 5. Version Compatibility

- Open Core and Enterprise interface versions should stay synchronized
- When modifying interfaces, update implementations in both repositories
- Use semantic versioning

### 6. Environment Isolation

- Development environment uses Open Core implementation
- Testing environment uses Enterprise implementation
- Control loading behavior through environment variables

## Common Issues

### Q1: Enterprise implementation not taking effect?

**Check**:
1. **Confirm both packages are installed in the same virtual environment** (most common issue!)
   - Run `cd memsys_opensource && uv pip list | grep memsys`
   - Should see both memsys and memsys-enterprise
2. Confirm both packages are editable installs (`uv pip install -e .`)
3. Verify entry points are registered successfully
4. Check interface name and implementation class name are consistent
5. Confirm `@Component()` decorator is added
6. Check addon scan paths include the implementation

### Q0: Cannot find core module?

**Error message**: `ModuleNotFoundError: No module named 'core'`

**Reason**: Enterprise is not installed in the opensource virtual environment.

**Solution**:
```bash
# In enterprise directory, install using opensource pip
cd ~/workspace/memsys_enterprise
../memsys_opensource/.venv/bin/pip install -e .

# Verify installation
cd ~/workspace/memsys_opensource
uv pip list | grep memsys-enterprise
```

### Q2: How to debug addon loading?

```python
# Method 1: Check logs
export LOG_LEVEL=DEBUG
uv run python -m src.run

# Method 2: Print in code
from core.addons.addons_registry import ADDONS_REGISTRY
print(f"Loaded {ADDONS_REGISTRY.count()} addons")
for addon in ADDONS_REGISTRY.get_all():
    print(f"  - {addon.name}")

# Method 3: Use Python debugger
import ipdb; ipdb.set_trace()
```

### Q3: Can there be multiple Enterprise addons?

Yes! You can create multiple addon packages:

```toml
# memsys_enterprise/pyproject.toml
[project.entry-points."memsys.addons"]
enterprise = "memsys_enterprise.addon"

# memsys_plugin_xyz/pyproject.toml
[project.entry-points."memsys.addons"]
plugin_xyz = "memsys_plugin_xyz.addon"
```

All addons will be loaded, following the same priority mechanism.

### Q4: How to disable an addon during local development?

Use the `MEMSYS_ENTRYPOINTS_FILTER` environment variable:

```bash
# Load only core, not enterprise
export MEMSYS_ENTRYPOINTS_FILTER=core

# Load core and plugin_xyz, not enterprise
export MEMSYS_ENTRYPOINTS_FILTER=core,plugin_xyz
```

### Q5: How to collaborate on code between two repositories?

Recommended workflow:

1. **Interface changes**: Modify interfaces in Open Core, submit PR
2. **Implementation updates**: After interfaces merge, update implementations in both repositories
3. **Version sync**: Ensure interface version numbers are consistent in both repositories
4. **Integration testing**: Install both packages locally for testing

### Q6: How to deploy in production?

```bash
# Method 1: Install published packages
pip install memsys
pip install memsys-enterprise

# Method 2: Install from source
pip install /path/to/memsys_opensource
pip install /path/to/memsys_enterprise

# Method 3: Use Docker
# Install both packages in Dockerfile
RUN pip install memsys memsys-enterprise
```

All methods have the same effect - entry points are automatically registered and loaded.

## Advanced Topics

### 1. Dependencies Between Addons

While there are no hard dependencies between addons, they can collaborate through interfaces:

```python
# Open Core defines two interfaces
class ServiceA(ABC): ...
class ServiceB(ABC): ...

# Enterprise implementation of ServiceA can use ServiceB
@Component()
class EnterpriseServiceA(ServiceA):
    def __init__(self):
        self.service_b: ServiceB = inject(ServiceB)
```

### 2. Addon Configuration

You can provide dedicated configuration for addons:

```python
# memsys_enterprise/src/memsys_enterprise/config/enterprise_config.py
from pydantic_settings import BaseSettings

class EnterpriseConfig(BaseSettings):
    mongodb_uri: str
    elasticsearch_url: str
    milvus_host: str
    
    class Config:
        env_prefix = "ENTERPRISE_"

# Use in implementation
@Component()
class CloudMemoryRepository(MemoryRepository):
    def __init__(self):
        self.config = EnterpriseConfig()
```

### 3. Conditional Loading

Conditionally load certain components based on environment or configuration:

```python
# memsys_enterprise/src/memsys_enterprise/addon.py
import os

# Only load certain paths in production environment
if os.getenv("ENV") == "production":
    di_registry.add_scan_path(
        os.path.join(enterprise_base_path, "production_only")
    )
```

## Summary

Core workflow for addon development:

1. ✅ **Environment setup**: Clone both repositories to the same directory, **install both packages in the same virtual environment**
2. ✅ **Interface abstraction**: Define clear interfaces in Open Core
3. ✅ **Separate implementation**: Implement different versions in both repositories
4. ✅ **Automatic loading**: Automatically discover and load through Entry Points
5. ✅ **Priority replacement**: Enterprise implementations automatically override Core implementations

This architecture achieves:
- Code isolation (separation of open-source and commercial code)
- Seamless integration (transparent switching for users)
- Flexible extension (supports multiple addon packages)
- Easy maintenance (consistent directory structure)

Follow this guide to quickly develop and deploy MemSys addons, achieving flexible feature composition and commercialization.
