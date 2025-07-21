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
from app.api import user as user_api
from app.db.chat import db
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    """
    logger.info("Starting up...")
    await db.create_db_and_tables()
    logger.info("Database tables created or already exist.")
    yield
    logger.info("Shutting down...")


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
    return {"status": "healthy"}


app.include_router(user_api.router)
app.include_router(chat_api.router)
app.include_router(analysis_api.router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
