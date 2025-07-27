"""Extensible cache eviction policies for production use."""

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from ...utils.logger import logger
from .cache_metrics import cache_metrics


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    async def should_evict(self, entry: dict[str, Any]) -> bool:
        """Determine if an entry should be evicted."""
        pass

    @abstractmethod
    async def on_access(self, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Update entry metadata on access."""
        pass

    @abstractmethod
    async def on_evict(self, key: str, entry: dict[str, Any]) -> None:
        """Handle entry eviction."""
        pass

    @abstractmethod
    def get_eviction_candidates(
        self,
        entries: list[tuple[str, dict[str, Any]]],
        count: int,
    ) -> list[str]:
        """Get keys of entries to evict based on policy."""
        pass


class TTLEvictionPolicy(EvictionPolicy):
    """Time-to-live based eviction policy."""

    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl = default_ttl_seconds

    async def should_evict(self, entry: dict[str, Any]) -> bool:
        """Check if entry has expired."""
        created_at = entry.get("created_at")
        if not created_at:
            return True

        ttl = entry.get("ttl", self.default_ttl)

        try:
            created_time = datetime.fromisoformat(created_at)
            if created_time.tzinfo is None:
                created_time = created_time.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            return True

        age_seconds = (datetime.now(UTC) - created_time).total_seconds()

        return age_seconds > ttl

    async def on_access(self, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Update last accessed time."""
        entry["last_accessed"] = datetime.now(UTC).isoformat()
        entry["access_count"] = entry.get("access_count", 0) + 1
        return entry

    async def on_evict(self, key: str, entry: dict[str, Any]) -> None:
        """Record TTL eviction."""
        cache_metrics.record_eviction("semantic", "ttl_expired")
        logger.debug(f"Evicted expired entry: {key}")

    def get_eviction_candidates(
        self,
        entries: list[tuple[str, dict[str, Any]]],
        count: int,
    ) -> list[str]:
        """Get oldest entries for eviction."""
        sorted_entries = sorted(
            entries,
            key=lambda x: x[1].get("created_at", ""),
        )
        return [key for key, _ in sorted_entries[:count]]


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    async def should_evict(self, entry: dict[str, Any]) -> bool:
        """LRU doesn't evict based on entry state."""
        return False

    async def on_access(self, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Update access timestamp for LRU tracking."""
        entry["last_accessed"] = datetime.now(UTC).isoformat()
        entry["access_count"] = entry.get("access_count", 0) + 1
        return entry

    async def on_evict(self, key: str, entry: dict[str, Any]) -> None:
        """Record LRU eviction."""
        cache_metrics.record_eviction("semantic", "lru")
        logger.debug(f"LRU evicted entry: {key}")

    def get_eviction_candidates(
        self,
        entries: list[tuple[str, dict[str, Any]]],
        count: int,
    ) -> list[str]:
        """Get least recently used entries."""
        sorted_entries = sorted(
            entries,
            key=lambda x: x[1].get("last_accessed", x[1].get("created_at", "")),
        )
        return [key for key, _ in sorted_entries[:count]]


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    async def should_evict(self, entry: dict[str, Any]) -> bool:
        """LFU doesn't evict based on entry state."""
        return False

    async def on_access(self, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Increment access count for LFU tracking."""
        entry["last_accessed"] = datetime.now(UTC).isoformat()
        entry["access_count"] = entry.get("access_count", 0) + 1
        return entry

    async def on_evict(self, key: str, entry: dict[str, Any]) -> None:
        """Record LFU eviction."""
        cache_metrics.record_eviction("semantic", "lfu")
        logger.debug(f"LFU evicted entry: {key}")

    def get_eviction_candidates(
        self,
        entries: list[tuple[str, dict[str, Any]]],
        count: int,
    ) -> list[str]:
        """Get least frequently used entries."""
        sorted_entries = sorted(
            entries,
            key=lambda x: (
                x[1].get("access_count", 0),
                x[1].get("created_at", ""),
            ),
        )
        return [key for key, _ in sorted_entries[:count]]


class HybridEvictionPolicy(EvictionPolicy):
    """
    Hybrid policy combining TTL and LRU.
    Evicts expired entries first, then uses LRU.
    """

    def __init__(
        self,
        ttl_policy: TTLEvictionPolicy | list[EvictionPolicy],
        lru_policy: LRUEvictionPolicy | list[EvictionPolicy] | None = None,
    ):
        if isinstance(ttl_policy, list):
            self.ttl_policies = ttl_policy
            self.lru_policies = lru_policy or []
            self.ttl_policy = ttl_policy[0] if ttl_policy else TTLEvictionPolicy()
            self.lru_policy = lru_policy[0] if lru_policy else LRUEvictionPolicy()
        else:
            self.ttl_policy = ttl_policy
            self.lru_policy = lru_policy
            self.ttl_policies = [ttl_policy]
            self.lru_policies = [lru_policy] if lru_policy else []

    async def should_evict(self, entry: dict[str, Any]) -> bool:
        """Check TTL expiration using any ttl policy that says evict."""
        for policy in self.ttl_policies:
            if await policy.should_evict(entry):
                return True
        return False

    async def on_access(self, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Update both TTL and LRU metadata."""
        for policy in self.ttl_policies:
            entry = await policy.on_access(key, entry)
        for policy in self.lru_policies:
            entry = await policy.on_access(key, entry)
        return entry

    async def on_evict(self, key: str, entry: dict[str, Any]) -> None:
        """Record eviction with appropriate reason."""
        if await self.should_evict(entry):
            await self.ttl_policy.on_evict(key, entry)
        else:
            await self.lru_policy.on_evict(key, entry)

    def get_eviction_candidates(
        self,
        entries: list[tuple[str, dict[str, Any]]],
        count: int,
    ) -> list[str]:
        """Get expired entries first, then LRU."""
        expired = []
        active = []

        for key, entry in entries:
            is_expired = False
            for policy in self.ttl_policies:
                if asyncio.run(policy.should_evict(entry)):
                    is_expired = True
                    break

            if is_expired:
                expired.append((key, entry))
            else:
                active.append((key, entry))

        evict_keys = [key for key, _ in expired]

        if len(evict_keys) < count and self.lru_policies:
            remaining = count - len(evict_keys)
            lru_keys = self.lru_policies[0].get_eviction_candidates(active, remaining)
            evict_keys.extend(lru_keys)

        return evict_keys[:count]


class EvictionPolicyFactory:
    """Factory for creating eviction policy instances."""

    _policies = {
        "ttl": TTLEvictionPolicy,
        "lru": LRUEvictionPolicy,
        "lfu": LFUEvictionPolicy,
    }

    @classmethod
    def create(cls, policy_name: str, **kwargs) -> EvictionPolicy:
        """Create an eviction policy by name."""
        if policy_name == "hybrid":
            ttl_kwargs = kwargs.pop("ttl_kwargs", {})
            lru_kwargs = kwargs.pop("lru_kwargs", {})
            return HybridEvictionPolicy(TTLEvictionPolicy(**ttl_kwargs), LRUEvictionPolicy(**lru_kwargs))

        if policy_name not in cls._policies:
            raise ValueError(f"Unknown eviction policy: {policy_name}")

        policy_class = cls._policies[policy_name]
        return policy_class(**kwargs)

    @classmethod
    def register(cls, name: str, policy_class: type[EvictionPolicy]) -> None:
        """Register a new eviction policy."""
        cls._policies[name] = policy_class
