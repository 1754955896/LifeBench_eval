"""
EverMemOS-specific resource tracker.

Extends DefaultTracker with EverMemOS's docker-compose storage measurement
(MongoDB / Elasticsearch / Milvus / Redis volumes) and record counts
(MongoDB collections / Elasticsearch indices).

EverMemOS storage architecture
------------------------------
  MongoDB          …  conversation meta, message records, extracted memories
  Elasticsearch    …  BM25 search index (semantic memories / event log)
  Milvus (etcd)    …  vector index metadata
  Milvus (minio)   …  vector index object storage
  Milvus (data)    …  vector index engine data
  Redis            …  accumulation buffer / cache

Volume sizes come from ``docker system df -v``; record counts are read
directly from MongoDB (db "memsys") and Elasticsearch (no auth) — both
are best-effort and cached, so the sampling thread never blocks: the
daemon/databases must be reachable, and failures degrade to empty/0.0.
MongoDB credentials fall back to the docker-compose defaults and can be
overridden via ``MONGODB_*`` environment variables.
"""

import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)

# Volume name suffixes from systems/EverMemOS_bz/docker-compose.yaml.
# docker compose prefixes them with the project name, so match by suffix.
_VOLUME_SUFFIXES = (
    "mongodb_data",
    "elasticsearch_data",
    "milvus_etcd_data",
    "milvus_minio_data",
    "milvus_data",
    "redis_data",
)

# Beanie collection names from src/infra_layer/adapters/out/persistence/document/memory/
_MONGO_COLLECTIONS = (
    "episodic_memories",
    "event_log_records",
    "semantic_memories",
    "conversation_metas",
    "memcells",
)

# Index aliases from src/infra_layer/adapters/out/search/elasticsearch/memory/
_ES_INDICES = ("episodic-memory", "event-log", "semantic-memory")

_MONGO_DEFAULTS = {
    "host": "localhost",
    "port": "27017",
    "username": "admin",
    "password": "memsys123",
    "database": "memsys",
}

# Longest units first so '1.5GB' matches GB, not the trailing 'B'.
_SIZE_UNITS = {"TB": 1e12, "GB": 1e9, "MB": 1e6, "kB": 1e3, "B": 1.0}

_STORAGE_TTL = 30.0        # cache successful `docker system df -v` results
_COUNTS_TTL = 30.0         # cache successful record-count reads
_RETRY_AFTER = 60.0        # wait before retrying after a backend failure


def _parse_size(text: str) -> float:
    """Parse a docker human-readable size ('1.5GB', '12.3MB', '0B') to bytes."""
    text = (text or "").strip()
    if not text or text == "N/A":
        return 0.0
    for unit, factor in _SIZE_UNITS.items():
        if text.endswith(unit):
            try:
                return float(text[: -len(unit)]) * factor
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


class _TTLCache:
    """Caches a producer's result for ``ttl`` seconds; after a failure,
    retries are throttled to ``retry_after`` seconds."""

    def __init__(self, ttl: float, retry_after: float):
        self.ttl = ttl
        self.retry_after = retry_after
        self._value: Optional[Any] = None
        self._ts: float = 0.0
        self._fail_ts: float = 0.0

    def get(self, producer) -> Optional[Any]:
        now = time.time()
        if self._value is not None and now - self._ts < self.ttl:
            return self._value
        if self._fail_ts and now - self._fail_ts < self.retry_after:
            return None
        try:
            value = producer()
        except Exception as exc:
            logger.debug("TTL cache producer failed: %s", exc)
            self._fail_ts = now
            return None
        self._value = value
        self._ts = now
        self._fail_ts = 0.0
        return value


@register_tracker("evermemos")
class EvermemosTracker(DefaultTracker):
    """EverMemOS-specific resource tracker.

    Adds docker volume storage breakdown and database record counts on
    top of ``DefaultTracker`` (process RSS/CPU + LLM token usage via the
    proxy).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        self._volume_cache = _TTLCache(_STORAGE_TTL, _RETRY_AFTER)
        self._counts_cache = _TTLCache(_COUNTS_TTL, _RETRY_AFTER)
        self._mongo_client = None
        self._es_client = None

    @property
    def system_name(self) -> str:
        return "evermemos"

    # -- docker volume sizes (best-effort, cached) ---------------------------

    def _get_volume_sizes(self) -> Dict[str, float]:
        """Return {volume_name: bytes} for EverMemOS docker volumes."""
        return self._volume_cache.get(self._read_volume_sizes) or {}

    def _read_volume_sizes(self) -> Dict[str, float]:
        result = subprocess.run(
            ["docker", "system", "df", "-v"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or f"docker exit code {result.returncode}"
            )
        return self._parse_docker_df(result.stdout)

    @staticmethod
    def _parse_docker_df(output: str) -> Dict[str, float]:
        """Extract named-volume sizes from ``docker system df -v`` output."""
        sizes: Dict[str, float] = {}
        in_table = False
        for line in output.splitlines():
            if "VOLUME NAME" in line:
                in_table = True
                continue
            if not in_table or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name, size_text = parts[0], parts[-1]
            if name.endswith(_VOLUME_SUFFIXES):
                sizes[name] = _parse_size(size_text)
        return sizes

    def _storage_breakdown(self) -> Dict[str, float]:
        """{volume_name: size_mb} for the six EverMemOS volumes."""
        return {
            name: round(size / (1024 * 1024), 3)
            for name, size in self._get_volume_sizes().items()
        }

    # -- record counts (best-effort, cached) ---------------------------------

    def _get_record_counts(self) -> Optional[Dict[str, Any]]:
        """Return MongoDB + Elasticsearch document counts, or ``None``
        when the databases are unreachable."""
        return self._counts_cache.get(self._read_record_counts)

    def _mongo_connection(self):
        if self._mongo_client is None:
            import pymongo

            cfg = {
                key: os.environ.get(f"MONGODB_{key.upper()}", default)
                for key, default in _MONGO_DEFAULTS.items()
            }
            self._mongo_client = pymongo.MongoClient(
                host=cfg["host"],
                port=int(cfg["port"]),
                username=cfg["username"],
                password=cfg["password"],
                authSource="admin",
                serverSelectionTimeoutMS=1000,
                connectTimeoutMS=1000,
            )
        return self._mongo_client

    def _es_connection(self):
        if self._es_client is None:
            from elasticsearch import Elasticsearch

            host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
            port = os.environ.get("ELASTICSEARCH_PORT", "19200")
            # max_retries=0: a dead database must fail fast, not retry 3x —
            # snapshot() runs on the GlobalMonitor sampling thread.
            self._es_client = Elasticsearch(
                hosts=[f"http://{host}:{port}"],
                request_timeout=2,
                max_retries=0,
            )
        return self._es_client

    def _read_record_counts(self) -> Dict[str, Any]:
        counts: Dict[str, Any] = {}

        try:
            mongo = self._mongo_connection()
            db = mongo[os.environ.get("MONGODB_DATABASE", _MONGO_DEFAULTS["database"])]
            counts["mongo"] = {
                name: db[name].count_documents({}, maxTimeMS=2000)
                for name in _MONGO_COLLECTIONS
            }
        except Exception as exc:
            logger.debug("EverMemOS mongo count failed: %s", exc)

        try:
            es = self._es_connection()
            counts["elasticsearch"] = {
                index: es.count(index=index)["count"]
                for index in _ES_INDICES
            }
        except Exception as exc:
            logger.debug("EverMemOS es count failed: %s", exc)

        if not counts:
            raise RuntimeError("neither MongoDB nor Elasticsearch reachable")
        return counts

    # -- main snapshot --------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Take a resource snapshot with EverMemOS storage layers.

        Returns a ``SystemSnapshot`` whose ``extra`` carries:
        * ``storage_breakdown`` — per-backend docker volume sizes (MB)
        * ``record_counts`` — MongoDB / Elasticsearch document counts
          (best-effort)
        * plus all parent ``DefaultTracker`` fields (RSS, CPU, LLM tokens)
        """
        snapshot = super().snapshot()

        breakdown = self._storage_breakdown()
        if breakdown:
            snapshot.extra["storage_breakdown"] = breakdown
            # Override DefaultTracker's 0.0 — evermemos has measurable on-disk state.
            snapshot.storage_mb = round(sum(breakdown.values()), 3)

        record_counts = self._get_record_counts()
        if record_counts:
            snapshot.extra["record_counts"] = record_counts

        return snapshot

    # -- backend summary ------------------------------------------------------

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return EverMemOS backend summary (storage + record counts)."""
        stats = super().backend_specific_stats()
        breakdown = self._storage_breakdown()
        if breakdown:
            stats["storage_breakdown"] = breakdown
        record_counts = self._get_record_counts()
        if record_counts:
            stats["record_counts"] = record_counts
        return stats
