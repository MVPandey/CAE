"""Advanced tests for eviction policies."""

import asyncio
from datetime import UTC, datetime

import pytest

from app.services.cache.eviction_policies import (
    EvictionPolicy,
    EvictionPolicyFactory,
    HybridEvictionPolicy,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    TTLEvictionPolicy,
)


class CustomEvictionPolicy(EvictionPolicy):
    """Custom eviction policy for testing."""

    def __init__(self, always_evict: bool = False):
        self.always_evict = always_evict

    async def should_evict(self, entry: dict) -> bool:
        """Always return the configured eviction status."""
        return self.always_evict

    async def on_access(self, key: str, entry: dict) -> dict:
        """Mark entry as accessed."""
        entry["custom_accessed"] = True
        return entry

    async def on_evict(self, key: str, entry: dict) -> None:
        """Record custom eviction."""
        pass

    def get_eviction_candidates(self, entries: list[tuple[str, dict]], count: int) -> list[str]:
        """Return first N entries as candidates."""
        return [key for key, _ in entries[:count]]


class TestEvictionPolicyFactoryAdvanced:
    """Advanced tests for eviction policy factory."""

    def test_factory_registry_isolation(self):
        """Test that factory registry is properly isolated."""
        EvictionPolicyFactory.register("test_custom", CustomEvictionPolicy)

        policy = EvictionPolicyFactory.create("test_custom")
        assert isinstance(policy, CustomEvictionPolicy)

        if "test_custom" in EvictionPolicyFactory._policies:
            del EvictionPolicyFactory._policies["test_custom"]

    def test_factory_with_constructor_args(self):
        """Test creating policies with constructor arguments."""
        EvictionPolicyFactory.register("test_args", CustomEvictionPolicy)

        policy = EvictionPolicyFactory.create("test_args", always_evict=True)
        assert isinstance(policy, CustomEvictionPolicy)
        assert policy.always_evict is True

        if "test_args" in EvictionPolicyFactory._policies:
            del EvictionPolicyFactory._policies["test_args"]

    def test_factory_register_override(self):
        """Test overriding existing policy registration."""
        EvictionPolicyFactory.register("override_test", TTLEvictionPolicy)
        policy1 = EvictionPolicyFactory.create("override_test")
        assert isinstance(policy1, TTLEvictionPolicy)

        EvictionPolicyFactory.register("override_test", LRUEvictionPolicy)
        policy2 = EvictionPolicyFactory.create("override_test")
        assert isinstance(policy2, LRUEvictionPolicy)

        if "override_test" in EvictionPolicyFactory._policies:
            del EvictionPolicyFactory._policies["override_test"]

    def test_factory_invalid_policy_class(self):
        """Test registering invalid policy class."""

        class NotAPolicy:
            pass

        EvictionPolicyFactory.register("invalid", NotAPolicy)

        invalid_policy = EvictionPolicyFactory.create("invalid")
        assert isinstance(invalid_policy, NotAPolicy)

        with pytest.raises(AttributeError):
            invalid_policy.should_evict

        if "invalid" in EvictionPolicyFactory._policies:
            del EvictionPolicyFactory._policies["invalid"]


class TestHybridEvictionPolicyAdvanced:
    """Advanced tests for hybrid eviction policy."""

    @pytest.mark.asyncio
    async def test_hybrid_default_policies(self):
        """Test hybrid policy with default sub-policies."""
        ttl_policy = TTLEvictionPolicy()
        lru_policy = LRUEvictionPolicy()
        policy = HybridEvictionPolicy(ttl_policy, lru_policy)

        from datetime import UTC, datetime

        current_time = datetime.now(UTC).isoformat()
        entry = {"created_at": current_time, "ttl": 3600}
        assert await policy.should_evict(entry) is False

        updated_entry = await policy.on_access("key", entry)
        assert "last_accessed" in updated_entry
        assert "access_count" in updated_entry

        await policy.on_evict("key", {})  # Should not raise

    @pytest.mark.asyncio
    async def test_hybrid_single_policy(self):
        """Test hybrid policy with single sub-policy."""
        ttl_policy = TTLEvictionPolicy(default_ttl_seconds=60)
        policy = HybridEvictionPolicy([ttl_policy], [])

        old_entry = {"created_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat()}
        assert await policy.should_evict(old_entry) is True

        new_entry = {"created_at": datetime.now(UTC).isoformat()}
        assert await policy.should_evict(new_entry) is False

    @pytest.mark.asyncio
    async def test_hybrid_custom_policies(self):
        """Test hybrid with custom policies."""
        custom1 = CustomEvictionPolicy(always_evict=True)
        custom2 = CustomEvictionPolicy(always_evict=False)

        policy = HybridEvictionPolicy([custom1], [custom2])

        assert await policy.should_evict({}) is True

    def test_hybrid_get_eviction_candidates(self):
        """Test hybrid policy candidate selection."""
        ttl_policy = TTLEvictionPolicy()
        lru_policy = LRUEvictionPolicy()
        policy = HybridEvictionPolicy([ttl_policy], [lru_policy])

        entries = [
            ("key1", {"created_at": "2020-01-01T00:00:00+00:00", "last_accessed": "2023-01-01T00:00:00"}),
            ("key2", {"created_at": "2023-01-01T00:00:00+00:00", "last_accessed": "2020-01-01T00:00:00"}),
            ("key3", {"created_at": "2022-01-01T00:00:00+00:00", "last_accessed": "2022-01-01T00:00:00"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)
        assert len(candidates) == 2
        assert candidates[0] == "key1"


class TestEvictionPolicyEdgeCases:
    """Test edge cases for eviction policies."""

    @pytest.mark.asyncio
    async def test_ttl_policy_invalid_dates(self):
        """Test TTL policy with invalid date formats."""
        policy = TTLEvictionPolicy()

        entry = {"created_at": "not-a-date"}
        assert await policy.should_evict(entry) is True  # Should evict invalid entries

        entry = {"created_at": "2023-01-01T00:00:00"}
        assert await policy.should_evict(entry) is True

    def test_lru_policy_missing_timestamps(self):
        """Test LRU policy with missing timestamp fields."""
        policy = LRUEvictionPolicy()

        entries = [
            ("key1", {}),  # No timestamps
            ("key2", {"last_accessed": "2023-01-01T00:00:00"}),
            ("key3", {"created_at": "2023-01-01T00:00:00"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)
        assert len(candidates) == 2
        assert "key1" in candidates  # Entry with no timestamps

    def test_lfu_policy_equal_access_counts(self):
        """Test LFU policy with many entries having same access count."""
        policy = LFUEvictionPolicy()

        entries = [
            ("key1", {"access_count": 5, "created_at": "2023-01-03T00:00:00"}),
            ("key2", {"access_count": 5, "created_at": "2023-01-01T00:00:00"}),
            ("key3", {"access_count": 5, "created_at": "2023-01-02T00:00:00"}),
            ("key4", {"access_count": 5, "created_at": "2023-01-04T00:00:00"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)
        assert candidates == ["key2", "key3"]  # Oldest entries

    @pytest.mark.asyncio
    async def test_policy_concurrent_access(self):
        """Test that policies handle concurrent access correctly."""
        policy = LRUEvictionPolicy()

        entry = {"access_count": 0}

        tasks = [policy.on_access("key1", entry.copy()) for _ in range(10)]

        results = await asyncio.gather(*tasks)

        for result in results:
            assert result["access_count"] == 1  # Each gets a copy
            assert "last_accessed" in result
