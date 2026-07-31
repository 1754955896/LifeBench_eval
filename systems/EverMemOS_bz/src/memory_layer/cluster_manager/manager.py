"""ClusterManager - Core component for automatic memcell clustering."""

import asyncio
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from memory_layer.cluster_manager.config import ClusterManagerConfig
from memory_layer.cluster_manager.storage import ClusterStorage, InMemoryClusterStorage
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Try to import vectorize service
try:
    from agentic_layer.vectorize_service import get_vectorize_service
    VECTORIZE_SERVICE_AVAILABLE = True
except ImportError:
    VECTORIZE_SERVICE_AVAILABLE = False
    logger.warning("Vectorize service not available, clustering will be limited")


class ClusterState:
    """Internal state for a single group's clustering."""
    
    def __init__(self):
        """Initialize empty cluster state."""
        self.event_ids: List[str] = []
        self.timestamps: List[float] = []
        self.vectors: List[np.ndarray] = []
        self.cluster_ids: List[str] = []
        self.eventid_to_cluster: Dict[str, str] = {}
        self.next_cluster_idx: int = 0
        
        # Centroid-based clustering state
        self.cluster_centroids: Dict[str, np.ndarray] = {}
        self.cluster_counts: Dict[str, int] = {}
        self.cluster_last_ts: Dict[str, Optional[float]] = {}
    
    def assign_new_cluster(self, event_id: str) -> str:
        """Assign a new cluster ID to an event.
        
        Args:
            event_id: Event identifier
        
        Returns:
            New cluster ID
        """
        cluster_id = f"cluster_{self.next_cluster_idx:03d}"
        self.next_cluster_idx += 1
        self.eventid_to_cluster[event_id] = cluster_id
        self.cluster_ids.append(cluster_id)
        return cluster_id
    
    def add_to_cluster(
        self,
        event_id: str,
        cluster_id: str,
        vector: np.ndarray,
        timestamp: Optional[float]
    ) -> None:
        """Add an event to an existing cluster.
        
        Args:
            event_id: Event identifier
            cluster_id: Cluster to add to
            vector: Event embedding vector
            timestamp: Event timestamp
        """
        self.eventid_to_cluster[event_id] = cluster_id
        self.cluster_ids.append(cluster_id)
        self._update_cluster_centroid(cluster_id, vector, timestamp)
    
    def _update_cluster_centroid(
        self,
        cluster_id: str,
        vector: np.ndarray,
        timestamp: Optional[float]
    ) -> None:
        """Update cluster centroid with new vector.
        
        Args:
            cluster_id: Cluster identifier
            vector: New vector to incorporate
            timestamp: Timestamp to update
        """
        if vector is None or vector.size == 0:
            if timestamp is not None:
                prev_ts = self.cluster_last_ts.get(cluster_id)
                self.cluster_last_ts[cluster_id] = max(prev_ts or timestamp, timestamp)
            return
        
        count = self.cluster_counts.get(cluster_id, 0)
        if count <= 0:
            self.cluster_centroids[cluster_id] = vector.astype(np.float32, copy=False)
            self.cluster_counts[cluster_id] = 1
        else:
            current_centroid = self.cluster_centroids[cluster_id]
            if current_centroid.dtype != np.float32:
                current_centroid = current_centroid.astype(np.float32)
            new_centroid = (current_centroid * float(count) + vector) / float(count + 1)
            self.cluster_centroids[cluster_id] = new_centroid.astype(np.float32, copy=False)
            self.cluster_counts[cluster_id] = count + 1
        
        if timestamp is not None:
            prev_ts = self.cluster_last_ts.get(cluster_id)
            self.cluster_last_ts[cluster_id] = max(prev_ts or timestamp, timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "event_ids": self.event_ids,
            "timestamps": self.timestamps,
            "cluster_ids": self.cluster_ids,
            "eventid_to_cluster": self.eventid_to_cluster,
            "next_cluster_idx": self.next_cluster_idx,
            "cluster_centroids": {
                cid: centroid.tolist()
                for cid, centroid in self.cluster_centroids.items()
            },
            "cluster_counts": self.cluster_counts,
            "cluster_last_ts": self.cluster_last_ts,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClusterState":
        """Create ClusterState from dictionary.
        
        Args:
            data: Dictionary representation
        
        Returns:
            ClusterState instance
        """
        state = ClusterState()
        state.event_ids = list(data.get("event_ids", []))
        state.timestamps = list(data.get("timestamps", []))
        state.cluster_ids = list(data.get("cluster_ids", []))
        state.eventid_to_cluster = dict(data.get("eventid_to_cluster", {}))
        state.next_cluster_idx = int(data.get("next_cluster_idx", 0))
        
        centroids = data.get("cluster_centroids", {}) or {}
        state.cluster_centroids = {
            k: np.array(v, dtype=np.float32) for k, v in centroids.items()
        }
        state.cluster_counts = {
            k: int(v) for k, v in (data.get("cluster_counts", {}) or {}).items()
        }
        state.cluster_last_ts = {
            k: float(v) for k, v in (data.get("cluster_last_ts", {}) or {}).items()
        }
        
        return state


class ClusterManager:
    """Automatic clustering manager with event notifications.
    
    ClusterManager handles incremental clustering of memcells based on semantic
    similarity (embeddings) and temporal proximity. It provides event hooks for
    downstream processing (e.g., ProfileManager).
    
    Key Features:
    - Incremental clustering using centroid-based or nearest-neighbor algorithms
    - Automatic embedding extraction via vectorize service
    - Event callbacks on cluster assignments
    - Persistent cluster state with version tracking
    - Flexible storage backends
    
    Example:
        ```python
        # Initialize
        config = ClusterManagerConfig(
            similarity_threshold=0.65,
            max_time_gap_days=7
        )
        cluster_mgr = ClusterManager(config)
        
        # Register callback
        def on_cluster(group_id, memcell, cluster_id):
            print(f"Memcell {memcell['event_id']} -> {cluster_id}")
        
        cluster_mgr.on_cluster_assigned(on_cluster)
        
        # Attach to extractor
        cluster_mgr.attach_to_extractor(memcell_extractor)
        ```
    """
    
    def __init__(
        self,
        config: Optional[ClusterManagerConfig] = None,
        storage: Optional[ClusterStorage] = None
    ):
        """Initialize ClusterManager.
        
        Args:
            config: Clustering configuration (uses defaults if None)
            storage: Cluster storage backend (uses InMemoryClusterStorage if None)
        """
        self.config = config or ClusterManagerConfig()
        
        # Initialize storage
        if storage is None:
            storage = InMemoryClusterStorage(
                enable_persistence=self.config.enable_persistence,
                persist_dir=Path(self.config.persist_dir) if self.config.persist_dir else None
            )
        self._storage = storage
        
        # Internal state
        self._states: Dict[str, ClusterState] = {}
        self._callbacks: List[Callable] = []
        
        # Vectorize service
        self._vectorize_service = None
        if VECTORIZE_SERVICE_AVAILABLE:
            try:
                self._vectorize_service = get_vectorize_service()
            except Exception as e:
                logger.warning(f"Failed to initialize vectorize service: {e}")
        
        # Statistics
        self._stats = {
            "total_memcells": 0,
            "clustered_memcells": 0,
            "new_clusters": 0,
            "failed_embeddings": 0,
        }
    
    def on_cluster_assigned(self, callback: Callable[[str, Dict[str, Any], str], None]) -> None:
        """Register a callback for cluster assignment events.
        
        Callback signature:
            callback(group_id: str, memcell: Dict[str, Any], cluster_id: str) -> None
        
        Args:
            callback: Function to call when a memcell is assigned to a cluster
        """
        self._callbacks.append(callback)
    
    async def cluster_memcell(
        self,
        group_id: str,
        memcell: Dict[str, Any]
    ) -> Optional[str]:
        """Cluster a memcell and return its cluster ID.
        
        Args:
            group_id: Group/conversation identifier
            memcell: Memcell dictionary with event_id, timestamp, episode/summary
        
        Returns:
            Cluster ID if successful, None otherwise
        """
        self._stats["total_memcells"] += 1
        
        # Get or load state for this group
        if group_id not in self._states:
            # Try to load from storage first
            stored_state = await self._storage.load_cluster_state(group_id)
            if stored_state:
                # Restore state from storage
                self._states[group_id] = ClusterState.from_dict(stored_state)
                logger.debug(f"Loaded cluster state for group {group_id} from storage")
            else:
                # Create new state
                self._states[group_id] = ClusterState()
                logger.debug(f"Created new cluster state for group {group_id}")
        
        state = self._states[group_id]
        
        # Extract key fields
        event_id = str(memcell.get("event_id", ""))
        if not event_id:
            logger.warning("Memcell missing event_id, skipping clustering")
            return None
        
        timestamp = self._parse_timestamp(memcell.get("timestamp"))
        text = self._extract_text(memcell)
        
        # Get embedding
        vector = await self._get_embedding(text)
        if vector is None or vector.size == 0:
            logger.warning(f"Failed to get embedding for event {event_id}, creating singleton cluster")
            cluster_id = state.assign_new_cluster(event_id)
            state.event_ids.append(event_id)
            state.timestamps.append(timestamp or 0.0)
            state.vectors.append(np.zeros((1,), dtype=np.float32))
            self._stats["new_clusters"] += 1
            self._stats["failed_embeddings"] += 1
            await self._notify_callbacks(group_id, memcell, cluster_id)
            return cluster_id
        
        # Find best matching cluster
        cluster_id = self._find_best_cluster(state, vector, timestamp)
        
        # Add to cluster
        if cluster_id is None:
            # Create new cluster
            cluster_id = state.assign_new_cluster(event_id)
            state._update_cluster_centroid(cluster_id, vector, timestamp)
            self._stats["new_clusters"] += 1
        else:
            # Add to existing cluster
            state.add_to_cluster(event_id, cluster_id, vector, timestamp)
        
        # Update state
        state.event_ids.append(event_id)
        state.timestamps.append(timestamp or 0.0)
        state.vectors.append(vector)
        
        self._stats["clustered_memcells"] += 1
        
        # Save state to storage
        await self._storage.save_cluster_state(group_id, state.to_dict())
        
        # Notify callbacks
        await self._notify_callbacks(group_id, memcell, cluster_id)
        
        return cluster_id
    
    def _find_best_cluster(
        self,
        state: ClusterState,
        vector: np.ndarray,
        timestamp: Optional[float]
    ) -> Optional[str]:
        """Find the best matching cluster for a vector.
        
        Args:
            state: Cluster state for the group
            vector: Embedding vector to match
            timestamp: Timestamp of the event
        
        Returns:
            Cluster ID if match found, None to create new cluster
        """
        if not state.cluster_centroids:
            return None
        
        best_similarity = -1.0
        best_cluster_id = None
        
        vector_norm = np.linalg.norm(vector) + 1e-9
        
        for cluster_id, centroid in state.cluster_centroids.items():
            if centroid is None or centroid.size == 0:
                continue
            
            # Check time constraint
            if timestamp is not None:
                last_ts = state.cluster_last_ts.get(cluster_id)
                if last_ts is not None:
                    time_diff = abs(timestamp - last_ts)
                    if time_diff > self.config.max_time_gap_seconds:
                        continue
            
            # Compute cosine similarity
            centroid_norm = np.linalg.norm(centroid) + 1e-9
            similarity = float((centroid @ vector) / (centroid_norm * vector_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        
        # Check if best similarity meets threshold
        if best_similarity >= self.config.similarity_threshold:
            return best_cluster_id
        
        return None
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector if successful, None otherwise
        """
        if not self._vectorize_service:
            logger.warning("Vectorize service not available")
            return None
        
        try:
            vector_arr = await self._vectorize_service.get_embedding(text)
            if vector_arr is not None:
                return np.array(vector_arr, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
        
        return None
    
    def _extract_text(self, memcell: Dict[str, Any]) -> str:
        """Extract representative text from memcell.
        
        Priority: episode > summary > original_data
        
        Args:
            memcell: Memcell dictionary
        
        Returns:
            Extracted text
        """
        # Try episode first
        episode = memcell.get("episode")
        if isinstance(episode, str) and episode.strip():
            return episode.strip()
        
        # Try summary
        summary = memcell.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        
        # Fallback to compact original_data
        lines = []
        original_data = memcell.get("original_data")
        if isinstance(original_data, list):
            for item in original_data[:6]:  # Limit to first 6 messages
                if isinstance(item, dict):
                    content = item.get("content") or item.get("summary")
                    if content:
                        text = str(content).strip()
                        if text:
                            lines.append(text)
        
        return "\n".join(lines) if lines else str(memcell.get("event_id", ""))
    
    def _parse_timestamp(self, timestamp: Any) -> Optional[float]:
        """Parse timestamp to float seconds.
        
        Args:
            timestamp: Timestamp in various formats
        
        Returns:
            Timestamp in seconds if successful, None otherwise
        """
        if timestamp is None:
            return None
        
        try:
            if isinstance(timestamp, (int, float)):
                val = float(timestamp)
                # Convert milliseconds to seconds if needed
                if val > 10_000_000_000:
                    val = val / 1000.0
                return val
            elif isinstance(timestamp, str):
                from common_utils.datetime_utils import from_iso_format
                dt = from_iso_format(timestamp)
                return dt.timestamp()
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
        
        return None
    
    async def _notify_callbacks(
        self,
        group_id: str,
        memcell: Dict[str, Any],
        cluster_id: str
    ) -> None:
        """Notify all registered callbacks of cluster assignment.
        
        Args:
            group_id: Group identifier
            memcell: Memcell that was clustered
            cluster_id: Assigned cluster ID
        """
        for callback in self._callbacks:
            try:
                # Support both sync and async callbacks
                if asyncio.iscoroutinefunction(callback):
                    await callback(group_id, memcell, cluster_id)
                else:
                    callback(group_id, memcell, cluster_id)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def get_cluster_assignments(self, group_id: str) -> Dict[str, str]:
        """Get event_id -> cluster_id mapping for a group.
        
        Args:
            group_id: Group identifier
        
        Returns:
            Dictionary mapping event_id to cluster_id
        """
        return await self._storage.get_cluster_assignments(group_id)
    
    async def export_clusters(self, output_dir: Path) -> int:
        """Export cluster assignments to JSON files.
        
        Args:
            output_dir: Directory to save cluster maps
        
        Returns:
            Number of groups exported
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        
        for group_id, state in self._states.items():
            try:
                # Save full state
                state_file = output_dir / f"cluster_state_{group_id}.json"
                import json
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state.to_dict(), f, ensure_ascii=False, indent=2, default=str)
                
                # Save assignments map
                assignments_file = output_dir / f"cluster_map_{group_id}.json"
                with open(assignments_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"assignments": state.eventid_to_cluster},
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                
                count += 1
            
            except Exception as e:
                logger.error(f"Failed to export clusters for group {group_id}: {e}")
        
        logger.info(f"Exported {count} cluster maps to {output_dir}")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            **self._stats,
            "total_groups": len(self._states),
            "total_clusters": sum(
                len(set(state.cluster_ids)) for state in self._states.values()
            ),
        }
    
    def attach_to_extractor(self, memcell_extractor: Any) -> None:
        """Attach ClusterManager to a MemCellExtractor.
        
        Creates a _cluster_worker attribute on the extractor for compatibility.
        
        Args:
            memcell_extractor: ConvMemCellExtractor instance
        """
        # Create a wrapper that uses ClusterManager
        class ClusterManagerWrapper:
            def __init__(self, cluster_mgr: ClusterManager):
                self._cluster_mgr = cluster_mgr
                self._states = {}  # For compatibility
            
            def submit(self, group_id: Optional[str], memcell: Dict[str, Any]) -> None:
                """Submit memcell for clustering."""
                gid = group_id or "__default__"
                
                # Run clustering asynchronously
                asyncio.create_task(
                    self._cluster_mgr.cluster_memcell(gid, memcell)
                )
            
            def stop(self) -> None:
                """Compatibility method."""
                pass
            
            async def dump_to_dir(self, output_dir: str) -> None:
                """Export clusters to directory."""
                await self._cluster_mgr.export_clusters(Path(output_dir))
            
            def get_assignments(self) -> Dict[str, Dict[str, str]]:
                """Get all cluster assignments."""
                return {
                    gid: state.eventid_to_cluster
                    for gid, state in self._cluster_mgr._states.items()
                }
        
        # Attach the cluster worker (create if not exists)
        memcell_extractor._cluster_worker = ClusterManagerWrapper(self)
        
        logger.info("ClusterManager successfully attached to MemCellExtractor")

