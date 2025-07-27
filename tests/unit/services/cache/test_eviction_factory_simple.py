"""Simple tests for eviction policy factory coverage."""

from app.services.cache.eviction_policies import (
    EvictionPolicyFactory,
    HybridEvictionPolicy,
)


class TestEvictionFactorySimple:
    """Simple tests for eviction factory."""

    def test_create_hybrid_policy(self):
        """Test creating hybrid eviction policy via factory."""
        ttl_kwargs = {"default_ttl_seconds": 7200}
        policy = EvictionPolicyFactory.create("hybrid", ttl_kwargs=ttl_kwargs)

        assert isinstance(policy, HybridEvictionPolicy)
        assert policy.ttl_policy.default_ttl == 7200
        assert policy.lru_policy is not None

    def test_create_hybrid_with_all_options(self):
        """Test creating hybrid policy with all configuration options."""
        ttl_kwargs = {"default_ttl_seconds": 3600}
        lru_kwargs = {}  # LRU doesn't take constructor args

        policy = EvictionPolicyFactory.create("hybrid", ttl_kwargs=ttl_kwargs, lru_kwargs=lru_kwargs)

        assert isinstance(policy, HybridEvictionPolicy)
        assert policy.ttl_policy.default_ttl == 3600
