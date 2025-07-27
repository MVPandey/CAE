"""Tests for cache eviction policies."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.cache.eviction_policies import (
    EvictionPolicyFactory,
    HybridEvictionPolicy,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    TTLEvictionPolicy,
)


class TestTTLEvictionPolicy:
    """Test TTL eviction policy."""

    @pytest.mark.asyncio
    async def test_should_evict_expired(self):
        """Test that expired entries are marked for eviction."""
        policy = TTLEvictionPolicy(default_ttl_seconds=60)

        entry = {
            "created_at": datetime(2023, 1, 1, tzinfo=UTC).isoformat(),
            "ttl": 60,
        }

        assert await policy.should_evict(entry) is True

    @pytest.mark.asyncio
    async def test_should_evict_not_expired(self):
        """Test that non-expired entries are not marked for eviction."""
        policy = TTLEvictionPolicy(default_ttl_seconds=3600)

        entry = {
            "created_at": datetime.now(UTC).isoformat(),
            "ttl": 3600,
        }

        assert await policy.should_evict(entry) is False

    @pytest.mark.asyncio
    async def test_should_evict_no_created_at(self):
        """Test that entries without created_at are evicted."""
        policy = TTLEvictionPolicy()
        entry = {"ttl": 3600}

        assert await policy.should_evict(entry) is True

    @pytest.mark.asyncio
    async def test_on_access(self):
        """Test updating entry on access."""
        policy = TTLEvictionPolicy()
        entry = {"access_count": 5}

        updated = await policy.on_access("key1", entry)

        assert "last_accessed" in updated
        assert updated["access_count"] == 6

    @pytest.mark.asyncio
    @patch("app.services.cache.eviction_policies.cache_metrics")
    async def test_on_evict(self, mock_metrics):
        """Test eviction handling."""
        policy = TTLEvictionPolicy()

        await policy.on_evict("key1", {})

        mock_metrics.record_eviction.assert_called_once_with("semantic", "ttl_expired")

    def test_get_eviction_candidates(self):
        """Test getting eviction candidates."""
        policy = TTLEvictionPolicy()

        entries = [
            ("key1", {"created_at": "2023-01-01T00:00:00"}),
            ("key2", {"created_at": "2023-01-02T00:00:00"}),
            ("key3", {"created_at": "2023-01-03T00:00:00"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)

        assert candidates == ["key1", "key2"]


class TestLRUEvictionPolicy:
    """Test LRU eviction policy."""

    @pytest.mark.asyncio
    async def test_should_evict(self):
        """Test that LRU doesn't evict based on entry state."""
        policy = LRUEvictionPolicy()
        entry = {"any": "data"}

        assert await policy.should_evict(entry) is False

    @pytest.mark.asyncio
    async def test_on_access(self):
        """Test updating entry on access."""
        policy = LRUEvictionPolicy()
        entry = {"access_count": 10}

        updated = await policy.on_access("key1", entry)

        assert "last_accessed" in updated
        assert updated["access_count"] == 11

    @pytest.mark.asyncio
    @patch("app.services.cache.eviction_policies.cache_metrics")
    async def test_on_evict(self, mock_metrics):
        """Test eviction handling."""
        policy = LRUEvictionPolicy()

        await policy.on_evict("key1", {})

        mock_metrics.record_eviction.assert_called_once_with("semantic", "lru")

    def test_get_eviction_candidates(self):
        """Test getting eviction candidates."""
        policy = LRUEvictionPolicy()

        entries = [
            ("key1", {"last_accessed": "2023-01-01T00:00:00"}),
            ("key2", {"last_accessed": "2023-01-03T00:00:00"}),
            ("key3", {"last_accessed": "2023-01-02T00:00:00"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)

        assert candidates == ["key1", "key3"]

    def test_get_eviction_candidates_with_created_at_fallback(self):
        """Test eviction candidates using created_at as fallback."""
        policy = LRUEvictionPolicy()

        entries = [
            ("key1", {"created_at": "2023-01-01T00:00:00"}),
            ("key2", {"last_accessed": "2023-01-03T00:00:00"}),
            ("key3", {"created_at": "2023-01-02T00:00:00"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)

        assert candidates == ["key1", "key3"]


class TestLFUEvictionPolicy:
    """Test LFU eviction policy."""

    @pytest.mark.asyncio
    async def test_should_evict(self):
        """Test that LFU doesn't evict based on entry state."""
        policy = LFUEvictionPolicy()
        entry = {"any": "data"}

        assert await policy.should_evict(entry) is False

    @pytest.mark.asyncio
    async def test_on_access(self):
        """Test updating entry on access."""
        policy = LFUEvictionPolicy()
        entry = {"access_count": 3}

        updated = await policy.on_access("key1", entry)

        assert "last_accessed" in updated
        assert updated["access_count"] == 4

    @pytest.mark.asyncio
    @patch("app.services.cache.eviction_policies.cache_metrics")
    async def test_on_evict(self, mock_metrics):
        """Test eviction handling."""
        policy = LFUEvictionPolicy()

        await policy.on_evict("key1", {})

        mock_metrics.record_eviction.assert_called_once_with("semantic", "lfu")

    def test_get_eviction_candidates(self):
        """Test getting eviction candidates."""
        policy = LFUEvictionPolicy()

        entries = [
            ("key1", {"access_count": 5, "created_at": "2023-01-01"}),
            ("key2", {"access_count": 2, "created_at": "2023-01-02"}),
            ("key3", {"access_count": 5, "created_at": "2023-01-03"}),
            ("key4", {"access_count": 1, "created_at": "2023-01-04"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)

        assert candidates == ["key4", "key2"]

    def test_get_eviction_candidates_tie_breaker(self):
        """Test eviction candidates with same access count uses age as tie breaker."""
        policy = LFUEvictionPolicy()

        entries = [
            ("key1", {"access_count": 5, "created_at": "2023-01-03"}),
            ("key2", {"access_count": 5, "created_at": "2023-01-01"}),
            ("key3", {"access_count": 5, "created_at": "2023-01-02"}),
        ]

        candidates = policy.get_eviction_candidates(entries, 2)

        assert candidates == ["key2", "key3"]


class TestHybridEvictionPolicy:
    """Test hybrid eviction policy."""

    @pytest.mark.asyncio
    async def test_should_evict_delegates_to_ttl(self):
        """Test that should_evict delegates to TTL policy."""
        ttl_policy = MagicMock(spec=TTLEvictionPolicy)
        lru_policy = MagicMock(spec=LRUEvictionPolicy)
        ttl_policy.should_evict.return_value = True

        policy = HybridEvictionPolicy(ttl_policy, lru_policy)
        entry = {"created_at": "2023-01-01"}

        result = await policy.should_evict(entry)

        assert result is True
        ttl_policy.should_evict.assert_called_once_with(entry)

    @pytest.mark.asyncio
    async def test_on_access_calls_both_policies(self):
        """Test that on_access updates both policies."""
        ttl_policy = MagicMock(spec=TTLEvictionPolicy)
        lru_policy = MagicMock(spec=LRUEvictionPolicy)

        ttl_policy.on_access.return_value = {"step": "ttl"}
        lru_policy.on_access.return_value = {"step": "lru", "final": True}

        policy = HybridEvictionPolicy(ttl_policy, lru_policy)

        result = await policy.on_access("key1", {})

        assert result == {"step": "lru", "final": True}
        ttl_policy.on_access.assert_called_once()
        lru_policy.on_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_evict_ttl_expired(self):
        """Test eviction handling for TTL expired entries."""
        ttl_policy = MagicMock(spec=TTLEvictionPolicy)
        lru_policy = MagicMock(spec=LRUEvictionPolicy)

        async def should_evict_true(entry):
            return True

        ttl_policy.should_evict = should_evict_true

        policy = HybridEvictionPolicy(ttl_policy, lru_policy)
        entry = {"created_at": "2023-01-01"}

        await policy.on_evict("key1", entry)

        ttl_policy.on_evict.assert_called_once_with("key1", entry)
        lru_policy.on_evict.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_evict_lru(self):
        """Test eviction handling for LRU entries."""
        ttl_policy = MagicMock(spec=TTLEvictionPolicy)
        lru_policy = MagicMock(spec=LRUEvictionPolicy)

        async def should_evict_false(entry):
            return False

        ttl_policy.should_evict = should_evict_false

        policy = HybridEvictionPolicy(ttl_policy, lru_policy)
        entry = {"created_at": datetime.now(UTC).isoformat()}

        await policy.on_evict("key1", entry)

        ttl_policy.on_evict.assert_not_called()
        lru_policy.on_evict.assert_called_once_with("key1", entry)


class TestEvictionPolicyFactory:
    """Test eviction policy factory."""

    def test_create_ttl_policy(self):
        """Test creating TTL policy."""
        policy = EvictionPolicyFactory.create("ttl", default_ttl_seconds=7200)

        assert isinstance(policy, TTLEvictionPolicy)
        assert policy.default_ttl == 7200

    def test_create_lru_policy(self):
        """Test creating LRU policy."""
        policy = EvictionPolicyFactory.create("lru")

        assert isinstance(policy, LRUEvictionPolicy)

    def test_create_lfu_policy(self):
        """Test creating LFU policy."""
        policy = EvictionPolicyFactory.create("lfu")

        assert isinstance(policy, LFUEvictionPolicy)

    def test_create_unknown_policy(self):
        """Test creating unknown policy raises error."""
        with pytest.raises(ValueError, match="Unknown eviction policy: unknown"):
            EvictionPolicyFactory.create("unknown")

    def test_register_custom_policy(self):
        """Test registering custom policy."""

        class CustomPolicy(TTLEvictionPolicy):
            pass

        EvictionPolicyFactory.register("custom", CustomPolicy)

        policy = EvictionPolicyFactory.create("custom")
        assert isinstance(policy, CustomPolicy)
