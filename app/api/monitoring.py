"""Monitoring endpoints for system health and metrics."""

import time

from fastapi import APIRouter, Response

from ..services.cache.redis_manager import redis_manager
from ..services.cache.semantic_cache import semantic_cache
from ..services.embeddings.embedding_service import embedding_service
from ..utils.logger import logger
from ..utils.metrics import metrics_collector

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/cache/stats")
async def get_cache_statistics():
    """Get comprehensive cache performance statistics."""
    try:
        redis_info = await redis_manager.get_info()
        semantic_stats = semantic_cache.get_stats()
        embedding_stats = embedding_service.get_stats()

        return {
            "status": "healthy",
            "redis": redis_info,
            "semantic_cache": semantic_stats,
            "embeddings": embedding_stats,
            "recommendations": _get_cache_recommendations(semantic_stats),
        }
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/cache/health")
async def check_cache_health():
    """Quick health check for cache systems."""
    try:
        redis_healthy = await redis_manager.exists("health_check")

        return {
            "redis": "healthy" if redis_healthy or redis_manager._is_healthy else "unhealthy",
            "status": "healthy" if redis_healthy or redis_manager._is_healthy else "degraded",
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.delete("/cache/clear")
async def clear_cache():
    """Clear all cache entries (admin operation)."""
    try:
        semantic_cleared = await semantic_cache.clear_all()

        embedding_cleared = await embedding_service.clear_cache()

        logger.info(
            "Cache cleared",
            extra={
                "semantic_entries": semantic_cleared,
                "embedding_entries": embedding_cleared,
            },
        )

        return {
            "status": "success",
            "semantic_entries_cleared": semantic_cleared,
            "embedding_entries_cleared": embedding_cleared,
            "total_cleared": semantic_cleared + embedding_cleared,
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def _get_cache_recommendations(stats: dict) -> list[str]:
    """Generate recommendations based on cache statistics."""
    recommendations = []

    if stats["hit_rate"] < 0.2:
        recommendations.append(
            "Low cache hit rate. Consider adjusting similarity threshold or warming cache with common patterns."
        )

    if stats["hit_rate"] > 0.9:
        recommendations.append("Very high cache hit rate. Consider reducing TTL to ensure fresh responses.")

    if stats["total_requests"] > 10000:
        recommendations.append(
            "High cache usage. Monitor memory consumption and consider implementing cache size limits."
        )

    return recommendations


@router.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format."""
    try:
        metrics_data = metrics_collector.get_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        return Response(content=f"# Error: {str(e)}", media_type="text/plain", status_code=500)


@router.get("/metrics/json")
async def get_metrics_json():
    """Get metrics in JSON format for easier consumption."""
    try:
        metrics_dict = metrics_collector.get_metrics_dict()
        return {
            "status": "success",
            "metrics": metrics_dict,
            "timestamp": int(time.time()),
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
