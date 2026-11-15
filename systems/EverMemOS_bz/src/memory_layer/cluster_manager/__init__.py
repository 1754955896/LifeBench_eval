"""Cluster Manager - Automatic clustering of memcells with event notifications.

This module provides ClusterManager, a core component that clusters memcells
based on semantic similarity and temporal proximity, with event hooks for
downstream processing.

Key Features:
- Incremental clustering using embeddings and timestamps
- Event notifications on cluster assignments
- Flexible storage backends for cluster state
- Seamless integration with MemCellExtractor

Usage:
    from memory_layer.cluster_manager import ClusterManager, ClusterManagerConfig
    
    # Initialize
    config = ClusterManagerConfig(
        similarity_threshold=0.65,
        max_time_gap_days=7,
        enable_persistence=True
    )
    cluster_mgr = ClusterManager(config)
    
    # Attach to memcell extractor
    cluster_mgr.attach_to_extractor(memcell_extractor)
    
    # Register callbacks for cluster events
    cluster_mgr.on_cluster_assigned(my_callback)
    
    # Clusters are automatically assigned, callbacks notified!
"""

from memory_layer.cluster_manager.config import ClusterManagerConfig
from memory_layer.cluster_manager.manager import ClusterManager
from memory_layer.cluster_manager.storage import (
    ClusterStorage,
    InMemoryClusterStorage,
)
from memory_layer.cluster_manager.mongo_cluster_storage import MongoClusterStorage

__all__ = [
    "ClusterManager",
    "ClusterManagerConfig",
    "ClusterStorage",
    "InMemoryClusterStorage",
    "MongoClusterStorage",
]

