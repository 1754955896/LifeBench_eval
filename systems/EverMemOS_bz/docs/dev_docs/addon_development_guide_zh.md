# 扩展开发指南

## 概述

本指南介绍如何为 MemSys 开发扩展（Addon），包括环境搭建、开发流程和最佳实践。以 Enterprise 扩展为例，说明如何实现商业功能与开源功能的代码切分。

## 快速开始（4 步启动）

如果你想快速启动扩展开发，只需按照以下 4 个步骤操作即可开始开发：

### 1. 克隆两个仓库到同一目录

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone <opensource-repo-url> memsys_opensource
git clone <enterprise-repo-url> memsys_enterprise
```

**重要**：两个仓库必须在同一个父目录下。

### 2. 在 opensource 创建虚拟环境并安装依赖

```bash
cd ~/workspace/memsys_opensource
uv sync
```

这会在 `memsys_opensource/.venv` 目录下创建虚拟环境并安装所有依赖。

### 3. 可编辑安装两个包到同一虚拟环境

**关键**：两个包都必须安装到 **opensource 的虚拟环境**中（因为 enterprise 需要导入 opensource 的模块）。

```bash
# 先安装 opensource
cd ~/workspace/memsys_opensource
source .venv/bin/activate
uv pip install -e .

# 再安装 enterprise 到同一个虚拟环境（重要！）
cd ~/workspace/memsys_enterprise
source ../memsys_opensource/.venv/bin/activate
uv pip install -e .
```

**为什么要这样做？**
- Enterprise 的代码需要 `from core.xxx import xxx` 导入 opensource 的模块
- 两个包必须在同一个 Python 环境中才能相互访问
- Entry points 必须在同一个环境中才能被系统发现

### 4. 启动服务

```bash
cd ~/workspace/memsys_opensource
uv run python -m src.run
```

如果看到以下日志，说明扩展加载成功：

```
🔌 开始加载 addons entry points...
  ✅ 已加载 entrypoint: core
  ✅ 已加载 entrypoint: enterprise
✅ Addons entry points 加载完成，共 2 个
```

现在你可以开始开发了！修改任何一个仓库的代码后，直接重启服务即可（无需重新安装）。

---

## 详细说明

下面是完整的环境准备和开发流程说明。

## 开发环境准备

### 1. 克隆代码仓库

首先需要将 opensource 和 enterprise 两个仓库克隆到**同一个目录**下：

```bash
# 创建工作目录
mkdir -p ~/workspace
cd ~/workspace

# 克隆 opensource 仓库
git clone <opensource-repo-url> memsys_opensource

# 克隆 enterprise 仓库
git clone <enterprise-repo-url> memsys_enterprise
```

**重要**：两个仓库必须放在同一个目录下，这是为了确保开发时的模块引用能够正确解析。

最终目录结构应该是：

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

### 2. 安装 opensource 依赖

进入 opensource 仓库，使用 uv 创建虚拟环境并安装依赖：

```bash
cd ~/workspace/memsys_opensource

# 使用 uv 同步依赖（会自动创建虚拟环境）
uv sync

# 或者如果已有虚拟环境
uv sync --frozen
```

这会在 `memsys_opensource/.venv` 目录下创建虚拟环境。

### 3. 可编辑安装两个包到同一虚拟环境

**重要**：opensource 和 enterprise 两个包必须安装到**同一个虚拟环境**中。

```bash
# 第一步：在 opensource 仓库根目录，安装 opensource
cd ~/workspace/memsys_opensource
uv pip install -e .

# 第二步：在 enterprise 仓库根目录，将 enterprise 也安装到 opensource 的虚拟环境
cd ~/workspace/memsys_enterprise

# 方式 1：使用 pip 直接安装（推荐）
../memsys_opensource/.venv/bin/pip install -e .

# 方式 2：使用 uv 指定 Python 解释器
uv pip install -e . --python ../memsys_opensource/.venv/bin/python

# 方式 3：先激活 opensource 的虚拟环境再安装
source ../memsys_opensource/.venv/bin/activate  # Linux/macOS
pip install -e .
```

**为什么必须在同一个虚拟环境中？**
- Enterprise 的代码需要 `from core.xxx import xxx` 导入 opensource 的模块
- 如果安装在不同的虚拟环境，enterprise 将无法找到 core 模块
- 两个包共享依赖，避免重复安装
- Entry points 必须在同一个环境中才能被正确发现

**可编辑安装的作用**：
- 代码修改后无需重新安装即可生效
- Entry points 会被注册到环境中
- 可以像正常安装的包一样导入

### 4. 验证安装

验证两个包是否正确安装到同一个虚拟环境中：

```bash
# 在 opensource 目录下检查已安装的包
cd ~/workspace/memsys_opensource
uv pip list | grep memsys

# 应该看到类似输出（注意都显示为可编辑安装）：
# memsys            0.1.0   /path/to/memsys_opensource/src
# memsys-enterprise 0.1.0   /path/to/memsys_enterprise/src/memsys_enterprise
```

验证 entry points 是否注册：

```bash
# 使用 opensource 的 Python 环境
cd ~/workspace/memsys_opensource

# 方式 1：使用 uv run
uv run python -c "
from importlib.metadata import entry_points
eps = entry_points(group='memsys.addons')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"

# 方式 2：激活虚拟环境后运行
source .venv/bin/activate
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='memsys.addons')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"

# 应该看到输出：
# core: src.addon
# enterprise: memsys_enterprise.addon
```

**如果没看到 enterprise**：说明 enterprise 没有安装到正确的虚拟环境中，请重新执行步骤 3。

### 5. 启动服务

通过 opensource 仓库启动服务，将会自动加载 enterprise 扩展：

```bash
cd ~/workspace/memsys_opensource

# 方式 1：使用 uv run（推荐）
uv run python -m src.run

# 方式 2：激活虚拟环境后运行
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
python -m src.run

# 方式 3：使用 project.scripts 定义的命令
uv run web
```

启动后，在日志中应该能看到类似输出：

```
🔌 开始加载 addons entry points...
  ✅ 已加载 entrypoint: core
  ✅ 已加载 entrypoint: enterprise
✅ Addons entry points 加载完成，共 2 个
```

## 扩展开发原理

### 核心思想

扩展本质上**不是**依赖关系，而是 Open Core 的一部分代码。扩展的目录结构和 Open Core 的目录结构完全一致，它们是**同一个系统的不同部分**。

### 关键机制

1. **接口抽象**：通过抽象类或协议定义接口
2. **分头实现**：在不同仓库中提供不同实现
3. **自动替换**：通过优先级机制，addon 的实现自动覆盖 open core 的实现

### 工作流程

```
┌─────────────────┐
│  定义接口抽象    │
│  (Open Core)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌───────────┐
│开源实现│  │  商业实现  │
│(Core) │  │(Enterprise)│
└───────┘  └───────────┘
    │         │
    └────┬────┘
         │
         ▼
   ┌─────────┐
   │优先级机制│
   │自动替换  │
   └─────────┘
```

## 开发扩展的步骤

### 步骤 1：接口抽象

当你需要对某一个功能或逻辑进行区分时，首先要进行接口抽象。

**在 Open Core 中定义接口**：

```python
# memsys_opensource/src/core/interface/repository/memory_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from core.domain.model.memory import Memory

class MemoryRepository(ABC):
    """
    记忆存储仓库接口
    定义记忆的 CRUD 操作规范
    """
    
    @abstractmethod
    async def save(self, memory: Memory) -> str:
        """
        保存记忆
        
        Args:
            memory: 记忆对象
            
        Returns:
            str: 记忆ID
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        根据ID查找记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            Optional[Memory]: 记忆对象，不存在则返回 None
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        搜索记忆
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            
        Returns:
            List[Memory]: 记忆列表
        """
        pass
```

### 步骤 2：Open Core 实现

在 Open Core 中提供基础实现（通常是简化版或本地版）。

```python
# memsys_opensource/src/infra_layer/adapters/out/persistence/repository/local_memory_repository.py
from typing import List, Optional
from core.interface.repository.memory_repository import MemoryRepository
from core.domain.model.memory import Memory
from core.di.component import Component

@Component()
class LocalMemoryRepository(MemoryRepository):
    """
    本地内存存储实现（用于开发和测试）
    数据存储在内存中，服务重启后丢失
    """
    
    def __init__(self):
        self._storage = {}  # 简单的内存字典存储
    
    async def save(self, memory: Memory) -> str:
        """保存到内存字典"""
        memory_id = memory.id or self._generate_id()
        self._storage[memory_id] = memory
        return memory_id
    
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """从内存字典查找"""
        return self._storage.get(memory_id)
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """简单的全文匹配搜索"""
        results = []
        for memory in self._storage.values():
            if query.lower() in memory.content.lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        return results
    
    def _generate_id(self) -> str:
        """生成简单的ID"""
        import uuid
        return str(uuid.uuid4())
```

**在 Open Core 的 addon 中注册扫描路径**：

```python
# memsys_opensource/src/addon.py
paths_registry.add_scan_path(
    os.path.join(get_base_scan_path(), "infra_layer/adapters/out/persistence")
)
```

### 步骤 3：Enterprise 实现

在 Enterprise 中提供商业级实现（通常是分布式、云原生版本）。

```python
# memsys_enterprise/src/memsys_enterprise/infra_layer/adapters/out/persistence/repository/cloud_memory_repository.py
from typing import List, Optional
from core.interface.repository.memory_repository import MemoryRepository
from core.domain.model.memory import Memory
from core.di.component import Component

@Component()
class CloudMemoryRepository(MemoryRepository):
    """
    云端分布式存储实现
    使用 MongoDB + Elasticsearch + Milvus 实现高可用存储和搜索
    """
    
    def __init__(
        self,
        mongo_client,      # 注入 MongoDB 客户端
        es_client,         # 注入 Elasticsearch 客户端
        milvus_client,     # 注入 Milvus 客户端
    ):
        self.mongo = mongo_client
        self.es = es_client
        self.milvus = milvus_client
    
    async def save(self, memory: Memory) -> str:
        """保存到分布式存储"""
        # 1. 保存到 MongoDB（主存储）
        memory_id = await self._save_to_mongo(memory)
        
        # 2. 索引到 Elasticsearch（全文搜索）
        await self._index_to_elasticsearch(memory_id, memory)
        
        # 3. 保存向量到 Milvus（向量搜索）
        await self._save_to_milvus(memory_id, memory)
        
        return memory_id
    
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """从 MongoDB 查询"""
        return await self._find_from_mongo(memory_id)
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """混合搜索：向量搜索 + 全文搜索 + 重排序"""
        # 1. 向量搜索（语义相似）
        vector_results = await self._vector_search(query, limit * 2)
        
        # 2. 全文搜索（关键词匹配）
        text_results = await self._text_search(query, limit * 2)
        
        # 3. 混合重排序
        final_results = self._rerank(vector_results, text_results, limit)
        
        return final_results
    
    # ... 其他私有方法实现 ...
```

**在 Enterprise 的 addon 中注册扫描路径**：

```python
# memsys_enterprise/src/memsys_enterprise/addon.py
di_registry.add_scan_path(
    os.path.join(enterprise_base_path, "infra_layer/adapters/out/persistence")
)
```

### 步骤 4：优先级机制

当两个仓库都提供相同接口的实现时，后加载的 addon（Enterprise）会自动覆盖先加载的（Core）。

**实现原理**：
1. DI 容器在扫描组件时，遇到相同接口的实现会进行替换
2. Enterprise addon 在 Core addon 之后加载
3. `CloudMemoryRepository` 会替换 `LocalMemoryRepository`

**使用时无需关心具体实现**：

```python
# 业务层代码（在 Open Core 或 Enterprise 中都一样）
from core.interface.repository.memory_repository import MemoryRepository
from core.di.injector import inject

class MemoryService:
    def __init__(self):
        # 自动注入，运行时决定使用哪个实现
        self.repository: MemoryRepository = inject(MemoryRepository)
    
    async def save_memory(self, content: str) -> str:
        memory = Memory(content=content)
        # 开发环境：使用 LocalMemoryRepository
        # 生产环境：使用 CloudMemoryRepository
        return await self.repository.save(memory)
```

## 目录结构规范

### Open Core 结构

```
memsys_opensource/
├── src/
│   ├── addon.py                      # Core addon 注册
│   ├── core/                         # 核心领域层
│   │   ├── interface/                # 接口定义（关键）
│   │   │   ├── repository/           # 仓库接口
│   │   │   ├── service/              # 服务接口
│   │   │   └── controller/           # 控制器接口
│   │   ├── domain/                   # 领域模型
│   │   ├── di/                       # 依赖注入
│   │   ├── addons/                   # Addon 机制
│   │   └── ...
│   ├── infra_layer/                  # 基础设施层
│   │   └── adapters/
│   │       ├── input/                # 输入适配器
│   │       └── out/                  # 输出适配器
│   │           └── persistence/      # 持久化实现
│   ├── agentic_layer/                # Agent 层
│   ├── biz_layer/                    # 业务层
│   └── component/                    # 通用组件
└── pyproject.toml
```

### Enterprise 结构（镜像 Open Core）

```
memsys_enterprise/
├── src/
│   └── memsys_enterprise/
│       ├── addon.py                  # Enterprise addon 注册
│       └── infra_layer/              # 基础设施层（与 Open Core 对应）
│           └── adapters/
│               ├── input/            # 输入适配器（商业版实现）
│               │   ├── api/          # 额外的 API
│               │   └── mcp/          # 额外的协议
│               └── out/              # 输出适配器（商业版实现）
│                   ├── persistence/  # 分布式持久化
│                   └── search/       # 高级搜索
└── pyproject.toml
```

**关键原则**：
- Enterprise 的目录结构**镜像** Open Core
- 只包含需要替换或新增的部分
- 保持层次结构一致，便于理解和维护

## 配置 Entry Points

### Open Core 配置

```toml
# memsys_opensource/pyproject.toml
[project]
name = "memsys"
version = "0.1.0"
# ... 其他配置 ...

[project.entry-points."memsys.addons"]
core = "src.addon"
```

### Enterprise 配置

```toml
# memsys_enterprise/pyproject.toml
[project]
name = "memsys-enterprise"
version = "0.1.0"
# ... 其他配置 ...

[project.entry-points."memsys.addons"]
enterprise = "memsys_enterprise.addon"
```

**注意**：
- Entry point group 名称必须是 `"memsys.addons"`
- Entry point 名称（如 `core`、`enterprise`）可以自定义
- Entry point 值指向包含注册代码的模块

## 开发工作流

### 1. 日常开发

```bash
# 1. 修改代码（Open Core 或 Enterprise）
vim memsys_opensource/src/infra_layer/...
vim memsys_enterprise/src/memsys_enterprise/infra_layer/...

# 2. 直接启动测试（无需重新安装）
cd memsys_opensource
uv run python -m src.run

# 3. 查看日志，确认扩展加载
# 应该看到 "已加载 entrypoint: enterprise"
```

### 2. 添加新的扩展功能

```bash
# 1. 在 Open Core 定义接口
vim memsys_opensource/src/core/interface/service/new_service.py

# 2. 在 Open Core 提供基础实现
vim memsys_opensource/src/component/new_service_impl.py

# 3. 在 Enterprise 提供商业实现
vim memsys_enterprise/src/memsys_enterprise/component/new_service_impl.py

# 4. 确保扫描路径已配置（如果需要新路径）
vim memsys_opensource/src/addon.py
vim memsys_enterprise/src/memsys_enterprise/addon.py

# 5. 启动测试
cd memsys_opensource
uv run python -m src.run
```

### 3. 调试扩展加载

如果发现扩展没有加载或组件没有替换，可以：

```python
# 在代码中添加调试输出
from core.addons.addons_registry import ADDONS_REGISTRY

# 查看所有已加载的 addons
all_addons = ADDONS_REGISTRY.get_all()
for addon in all_addons:
    print(f"Addon: {addon.name}")
    if addon.has_di():
        for path in addon.di.get_scan_paths():
            print(f"  DI Path: {path}")
```

或者设置日志级别为 DEBUG：

```bash
export LOG_LEVEL=DEBUG
uv run python -m src.run
```

### 4. 只加载 Open Core（不加载 Enterprise）

```bash
# 设置环境变量，只加载 core addon
export MEMSYS_ENTRYPOINTS_FILTER=core

# 启动服务
cd memsys_opensource
uv run python -m src.run

# 此时只会加载 Open Core 的实现，不会加载 Enterprise
```

## 最佳实践

### 1. 接口先行

- 在开发新功能前，先思考接口设计
- 接口应该足够抽象，不包含实现细节
- 接口定义放在 `core/interface/` 目录下

### 2. 保持目录结构一致

- Enterprise 的目录结构应该镜像 Open Core
- 便于快速定位对应的实现
- 降低维护成本

### 3. 文档和注释

- 接口定义必须有详细的 docstring
- 说明每个方法的用途、参数、返回值
- 标注哪些是开源实现，哪些是商业实现

### 4. 测试覆盖

- 为接口编写单元测试
- 测试应该对两种实现都有效
- 使用依赖注入，方便 mock 和测试

### 5. 版本兼容

- Open Core 和 Enterprise 的接口版本应该保持同步
- 修改接口时，同时更新两个仓库的实现
- 使用语义化版本控制

### 6. 环境隔离

- 开发环境使用 Open Core 实现
- 测试环境使用 Enterprise 实现
- 通过环境变量控制加载行为

## 常见问题

### Q1: Enterprise 实现没有生效？

**检查项**：
1. **确认两个包都安装到同一个虚拟环境**（最常见的问题！）
   - 运行 `cd memsys_opensource && uv pip list | grep memsys`
   - 应该同时看到 memsys 和 memsys-enterprise
2. 确认两个包都进行了可编辑安装（`uv pip install -e .`）
3. 验证 entry points 是否注册成功
4. 检查接口名称和实现类名是否一致
5. 确认 `@Component()` 装饰器是否添加
6. 查看 addon 扫描路径是否包含该实现

### Q0: 找不到 core 模块？

**错误信息**：`ModuleNotFoundError: No module named 'core'`

**原因**：Enterprise 没有安装到 opensource 的虚拟环境中。

**解决方法**：
```bash
# 在 enterprise 目录下，使用 opensource 的 pip 安装
cd ~/workspace/memsys_enterprise
../memsys_opensource/.venv/bin/pip install -e .

# 验证安装
cd ~/workspace/memsys_opensource
uv pip list | grep memsys-enterprise
```

### Q2: 如何调试扩展加载？

```python
# 方法 1：查看日志
export LOG_LEVEL=DEBUG
uv run python -m src.run

# 方法 2：在代码中打印
from core.addons.addons_registry import ADDONS_REGISTRY
print(f"已加载 {ADDONS_REGISTRY.count()} 个 addons")
for addon in ADDONS_REGISTRY.get_all():
    print(f"  - {addon.name}")

# 方法 3：使用 Python 调试器
import ipdb; ipdb.set_trace()
```

### Q3: 可以有多个 Enterprise 扩展吗？

可以！你可以创建多个扩展包：

```toml
# memsys_enterprise/pyproject.toml
[project.entry-points."memsys.addons"]
enterprise = "memsys_enterprise.addon"

# memsys_plugin_xyz/pyproject.toml
[project.entry-points."memsys.addons"]
plugin_xyz = "memsys_plugin_xyz.addon"
```

所有扩展都会被加载，遵循相同的优先级机制。

### Q4: 如何在本地开发时禁用某个扩展?

使用 `MEMSYS_ENTRYPOINTS_FILTER` 环境变量：

```bash
# 只加载 core，不加载 enterprise
export MEMSYS_ENTRYPOINTS_FILTER=core

# 加载 core 和 plugin_xyz，不加载 enterprise
export MEMSYS_ENTRYPOINTS_FILTER=core,plugin_xyz
```

### Q5: 两个仓库的代码如何协作开发？

建议工作流：

1. **接口变更**：在 Open Core 中修改接口，提交 PR
2. **实现更新**：接口合并后，分别在两个仓库更新实现
3. **同步版本**：确保接口版本号在两个仓库中一致
4. **集成测试**：在本地同时安装两个包进行测试

### Q6: 生产环境如何部署？

```bash
# 方法 1：安装发布的包
pip install memsys
pip install memsys-enterprise

# 方法 2：从源码安装
pip install /path/to/memsys_opensource
pip install /path/to/memsys_enterprise

# 方法 3：使用 Docker
# Dockerfile 中安装两个包
RUN pip install memsys memsys-enterprise
```

所有方法效果相同，entry points 会自动注册和加载。

## 进阶主题

### 1. 扩展之间的依赖

虽然扩展之间没有硬依赖，但可以通过接口进行协作：

```python
# Open Core 定义两个接口
class ServiceA(ABC): ...
class ServiceB(ABC): ...

# Enterprise 实现 ServiceA 时可以使用 ServiceB
@Component()
class EnterpriseServiceA(ServiceA):
    def __init__(self):
        self.service_b: ServiceB = inject(ServiceB)
```

### 2. 扩展配置

可以为扩展提供专门的配置：

```python
# memsys_enterprise/src/memsys_enterprise/config/enterprise_config.py
from pydantic_settings import BaseSettings

class EnterpriseConfig(BaseSettings):
    mongodb_uri: str
    elasticsearch_url: str
    milvus_host: str
    
    class Config:
        env_prefix = "ENTERPRISE_"

# 在实现中使用
@Component()
class CloudMemoryRepository(MemoryRepository):
    def __init__(self):
        self.config = EnterpriseConfig()
```

### 3. 条件性加载

根据环境或配置条件性地加载某些组件：

```python
# memsys_enterprise/src/memsys_enterprise/addon.py
import os

# 只在生产环境加载某些路径
if os.getenv("ENV") == "production":
    di_registry.add_scan_path(
        os.path.join(enterprise_base_path, "production_only")
    )
```

## 总结

扩展开发的核心流程：

1. ✅ **环境搭建**：克隆两个仓库到同一目录，**将两个包安装到同一虚拟环境**
2. ✅ **接口抽象**：在 Open Core 定义清晰的接口
3. ✅ **分头实现**：在两个仓库分别实现不同版本
4. ✅ **自动加载**：通过 Entry Points 自动发现和加载
5. ✅ **优先级替换**：Enterprise 实现自动覆盖 Core 实现

这种架构实现了：
- 代码隔离（开源和商业代码分离）
- 无缝集成（用户无感知切换）
- 灵活扩展（支持多个扩展包）
- 易于维护（目录结构一致）

遵循本指南，你可以快速开发和部署 MemSys 扩展，实现功能的灵活组合和商业化。

