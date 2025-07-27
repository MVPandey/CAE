import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import chat as chat_api
from app.api import conversation_analysis as analysis_api
from app.api import monitoring as monitoring_api
from app.api import user as user_api
from app.db.chat import db
from app.services.cache.redis_manager import redis_manager
from app.utils.config import app_settings
from app.utils.logger import logger
from app.utils.metrics import metrics_collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    """
    logger.info("Starting up...")
    
    # Log feature summary
    feature_summary = app_settings.get_feature_summary()
    logger.info(f"Feature configuration: {feature_summary}")

    await db.create_db_and_tables()
    logger.info("Database tables created or already exist.")

    await redis_manager.initialize()
    logger.info("Redis cache initialized.")

    # Conditionally initialize semantic cache
    if app_settings.enable_semantic_cache:
        try:
            from app.services.cache.semantic_cache import semantic_cache
            # semantic_cache will auto-initialize when imported if embedding keys are available
            logger.info("✅ Semantic caching enabled - embeddings available")
        except Exception as e:
            logger.warning(f"⚠️ Semantic caching disabled - initialization failed: {e}")
    else:
        logger.info("ℹ️ Semantic caching disabled - EMBEDDING_MODEL_API_KEY not provided")

    # Conditionally initialize metrics collector
    if app_settings.enable_prometheus_metrics:
        metrics_collector.initialize()
        logger.info("✅ Prometheus metrics enabled")
    else:
        logger.info("ℹ️ Prometheus metrics disabled (DISABLE_PROMETHEUS_METRICS=true)")

    yield

    logger.info("Shutting down...")
    await redis_manager.close()
    logger.info("Redis connections closed.")


app = FastAPI(
    title="CAE API",
    description="API for Conversational Analysis Engine",
    version="0.0.1",
    lifespan=lifespan,
    logger=logger,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their responses"""
    start_time = time.time()

    body = await request.body()
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "body": body.decode() if body else None,
        },
    )

    async def receive():
        return {"type": "http.request", "body": body}

    request._receive = receive

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {request.method} {request.url.path}",
        extra={
            "status_code": response.status_code,
            "process_time": f"{process_time:.3f}s",
        },
    )

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging"""
    error_dict = (
        {
            "errors": exc.errors(),
            "body": exc.body,
            "path": request.url.path,
            "method": request.method,
        },
    )
    logger.error(
        f"Validation error for {request.method} {request.url.path}: {error_dict}",
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "message": "Request validation failed. Check the 'detail' field for specific errors.",
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with logging"""
    logger.error(
        f"HTTP exception for {request.method} {request.url.path} | {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions"""
    logger.exception(
        f"Unhandled exception for {request.method} {request.url.path}",
        extra={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint that validates all services."""
    import time

    from .services.cache.redis_manager import redis_manager

    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": "0.0.1",
        "features": app_settings.get_feature_summary(),
        "services": {},
    }

    try:
        redis_healthy = redis_manager.is_healthy
        health_status["services"]["redis"] = {"status": "healthy" if redis_healthy else "unhealthy"}
        if not redis_healthy:
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    # Only check semantic cache if enabled
    if app_settings.enable_semantic_cache:
        try:
            from .services.cache.semantic_cache import semantic_cache
            cache_healthy = await semantic_cache.health_check()
            health_status["services"]["cache"] = {"status": "healthy" if cache_healthy else "unhealthy"}
            if not cache_healthy:
                health_status["status"] = "unhealthy"
                health_status["services"]["cache"]["error"] = "Cache health check failed"
        except Exception as e:
            health_status["services"]["cache"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "unhealthy"
    else:
        health_status["services"]["cache"] = {"status": "disabled", "reason": "EMBEDDING_MODEL_API_KEY not provided"}

    if health_status["status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_status)

    return health_status


@app.get("/health/detailed")
async def health_check_detailed():
    """Detailed health check with additional service information."""

    from .services.cache.redis_manager import redis_manager

    health_status = await health_check()
    if isinstance(health_status, JSONResponse):
        health_status = health_status.body.decode()
        import json

        health_status = json.loads(health_status)

    try:
        if hasattr(redis_manager, "get_connection_info"):
            health_status["services"]["redis"]["connection_info"] = redis_manager.get_connection_info()
    except Exception:
        pass

    # Only get semantic cache stats if enabled
    if app_settings.enable_semantic_cache:
        try:
            from .services.cache.semantic_cache import semantic_cache
            if hasattr(semantic_cache, "get_stats"):
                health_status["services"]["cache"]["stats"] = await semantic_cache.get_stats()
        except Exception:
            pass

    return health_status


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint with conditional response."""
    from fastapi import HTTPException, Response

    from .utils.metrics import metrics_collector

    if not app_settings.enable_prometheus_metrics:
        raise HTTPException(
            status_code=404, 
            detail="Metrics disabled. Set DISABLE_PROMETHEUS_METRICS=false to enable."
        )

    try:
        metrics_data = metrics_collector.get_metrics()
        return Response(content=metrics_data, media_type="text/plain; version=0.0.4")
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return Response(content=f"# Error: {str(e)}", media_type="text/plain; version=0.0.4", status_code=500)


@app.get("/metrics/json")
async def get_metrics_json():
    """JSON metrics endpoint with conditional response."""
    from fastapi import HTTPException
    
    from .utils.metrics import metrics_collector

    if not app_settings.enable_prometheus_metrics:
        raise HTTPException(
            status_code=404, 
            detail="Metrics disabled. Set DISABLE_PROMETHEUS_METRICS=false to enable."
        )

    try:
        metrics_dict = metrics_collector.get_metrics_dict()
        return metrics_dict
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


app.include_router(user_api.router)
app.include_router(chat_api.router)
app.include_router(analysis_api.router)
app.include_router(monitoring_api.router)


def run_uvicorn():
    """Run the application using uvicorn."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    run_uvicorn()
