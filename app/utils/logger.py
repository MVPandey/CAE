import json
import logging
import sys
from typing import Any

from loguru import logger

from .config import app_settings


class InterceptHandler(logging.Handler):
    def emit(self, record):
        if record.name == "uvicorn" and record.levelno < logging.INFO:
            return

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def format_record(record: dict[str, Any]) -> str:
    """Format log records with extra fields displayed inline."""

    base = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    extra = record.get("extra", {})
    if extra:
        display_extra = {k: v for k, v in extra.items() if not k.startswith("_")}

        if display_extra:
            extra_parts = []
            for key, value in display_extra.items():
                safe_key = str(key).replace("{", "{{").replace("}", "}}")

                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                elif isinstance(value, (dict, list)):
                    try:
                        value = json.dumps(value, separators=(",", ":"), default=str)
                        if len(value) > 100:
                            value = value[:97] + "..."
                    except Exception:
                        value = str(value)[:100]

                safe_value = str(value).replace("{", "{{").replace("}", "}}")

                extra_parts.append(f"<blue>{safe_key}</blue>=<yellow>{safe_value}</yellow>")

            base += " | " + " ".join(extra_parts)

    return base + "\n{exception}"


def format_record_json(record: dict[str, Any]) -> str:
    """Format log records as JSON for structured logging."""

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "module": record["module"],
    }

    extra = record.get("extra", {})
    for key, value in extra.items():
        if not key.startswith("_"):
            log_entry[key] = value

    if record["exception"]:
        exc_info = record["exception"]
        log_entry["exception"] = {
            "type": exc_info.type.__name__,
            "value": str(exc_info.value),
            "traceback": exc_info.traceback.raw if hasattr(exc_info.traceback, "raw") else str(exc_info.traceback),
        }

    return json.dumps(log_entry, default=str) + "\n"


logger.remove()

logger.add(
    sys.stderr,
    level=app_settings.LOG_LEVEL or "INFO",
    format=format_record,
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logger.add(
    "logs/app.log",
    rotation="00:00",
    retention="7 days",
    level=app_settings.LOG_LEVEL or "INFO",
    format=lambda r: format_record(r)
    .replace("<green>", "")
    .replace("</green>", "")
    .replace("<level>", "")
    .replace("</level>", "")
    .replace("<cyan>", "")
    .replace("</cyan>", "")
    .replace("<blue>", "")
    .replace("</blue>", "")
    .replace("<yellow>", "")
    .replace("</yellow>", ""),
    compression="zip",
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logger.add(
    "logs/app-structured.log",
    rotation="100 MB",
    retention="30 days",
    level="DEBUG",
    enqueue=True,
    serialize=True,
)

logger.add(
    "logs/errors.log",
    rotation="50 MB",
    retention="30 days",
    level="ERROR",
    format=lambda r: format_record(r)
    .replace("<green>", "")
    .replace("</green>", "")
    .replace("<level>", "")
    .replace("</level>", "")
    .replace("<cyan>", "")
    .replace("</cyan>", "")
    .replace("<blue>", "")
    .replace("</blue>", "")
    .replace("<yellow>", "")
    .replace("</yellow>", ""),
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler()]
    logging_logger.propagate = False

logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


class LoggerWrapper:
    """
    A wrapper class for loguru logger that adds support for 'extra' parameter
    without modifying the original logger instance.
    """

    def __init__(self, wrapped_logger):
        self._logger = wrapped_logger

    def _log_with_extra(self, method_name, message, *args, extra=None, **kwargs):
        """Helper method to handle logging with optional extra context."""
        if extra:
            bound_logger = self._logger.bind(**extra)
            getattr(bound_logger, method_name)(message, *args, **kwargs)
        else:
            getattr(self._logger, method_name)(message, *args, **kwargs)

    def info(self, message, *args, extra=None, **kwargs):
        self._log_with_extra("info", message, *args, extra=extra, **kwargs)

    def debug(self, message, *args, extra=None, **kwargs):
        self._log_with_extra("debug", message, *args, extra=extra, **kwargs)

    def warning(self, message, *args, extra=None, **kwargs):
        self._log_with_extra("warning", message, *args, extra=extra, **kwargs)

    def error(self, message, *args, extra=None, **kwargs):
        self._log_with_extra("error", message, *args, extra=extra, **kwargs)

    def critical(self, message, *args, extra=None, **kwargs):
        self._log_with_extra("critical", message, *args, extra=extra, **kwargs)

    def bind(self, **context):
        """Create a new logger instance with bound context."""
        return LoggerWrapper(self._logger.bind(**context))

    def opt(self, **options):
        """Pass through to the wrapped logger's opt method."""
        return self._logger.opt(**options)

    def __getattr__(self, name):
        """Delegate any other attributes/methods to the wrapped logger."""
        return getattr(self._logger, name)


def get_logger(**context):
    """
    Get a logger instance with bound context.

    Example:
        logger = get_logger(user_id="123", request_id="abc")
        logger.info("Processing request")  # Will include user_id and request_id

    Args:
        **context: Key-value pairs to bind to the logger

    Returns:
        LoggerWrapper instance with bound context
    """
    return LoggerWrapper(logger.bind(**context))


logger = LoggerWrapper(logger)

__all__ = ["logger", "get_logger"]
