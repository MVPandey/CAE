"""Tests to increase eviction factory coverage."""

from app.services.cache.eviction_policies import EvictionPolicyFactory, HybridEvictionPolicy


def test_create_hybrid_eviction_policy():
    """Test creating hybrid policy through factory."""
    policy = EvictionPolicyFactory.create("hybrid", default_ttl_seconds=3600)

    assert isinstance(policy, HybridEvictionPolicy)
    assert policy.ttl_policy is not None
    assert policy.lru_policy is not None
    assert policy.ttl_policy.default_ttl == 3600
