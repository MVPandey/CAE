"""Cache services for the Conversational Analysis Engine."""

from .cache_metrics import cache_metrics, track_cache_operation
from .eviction_policies import (
    EvictionPolicy,
    EvictionPolicyFactory,
    HybridEvictionPolicy,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    TTLEvictionPolicy,
)
from .redis_manager import redis_manager
from .semantic_cache import semantic_cache
from .similarity_strategies import (
    CosineSimilarityStrategy,
    DotProductSimilarityStrategy,
    EuclideanDistanceStrategy,
    HybridSimilarityStrategy,
    SimilarityStrategy,
    SimilarityStrategyFactory,
)

__all__ = [
    "redis_manager",
    "semantic_cache",
    "cache_metrics",
    "track_cache_operation",
    "EvictionPolicy",
    "TTLEvictionPolicy",
    "LRUEvictionPolicy",
    "LFUEvictionPolicy",
    "HybridEvictionPolicy",
    "EvictionPolicyFactory",
    "SimilarityStrategy",
    "CosineSimilarityStrategy",
    "EuclideanDistanceStrategy",
    "DotProductSimilarityStrategy",
    "HybridSimilarityStrategy",
    "SimilarityStrategyFactory",
]
