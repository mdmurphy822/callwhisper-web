"""
Structured Logging Configuration for CallWhisper

Based on LibV2 orchestrator-architecture course:
- "Never silently mask failures"
- "Include routing attempt history in response"
- "Metadata Enrichment"
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from functools import lru_cache

import structlog


def configure_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    json_output: bool = True
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_dir: Directory for log files (if None, logs to stdout only)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON; if False, output human-readable
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        stream=sys.stdout,
    )

    # Define processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for production/parsing
        final_processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Human-readable output for development
        final_processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=final_processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if directory provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"callwhisper_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


@lru_cache(maxsize=1)
def get_logger(name: str = "callwhisper") -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually module name)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary logging context.

    Usage:
        with LogContext(request_id="123", operation="transcribe"):
            logger.info("Starting operation")
            # All logs within this block will include request_id and operation
    """

    def __init__(self, **kwargs):
        self.context = kwargs

    def __enter__(self):
        for key, value in self.context.items():
            structlog.contextvars.bind_contextvars(**{key: value})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.unbind_contextvars(*self.context.keys())
        return False


def log_operation(
    logger: structlog.BoundLogger,
    operation: str,
    **context
):
    """
    Decorator/context for logging operation start/end.

    Based on LibV2 pattern: "Default handlers should clearly indicate
    in their response that default handling was invoked and provide
    metadata about why specialized routing failed."
    """
    class OperationLogger:
        def __init__(self):
            self.start_time = None

        def __enter__(self):
            self.start_time = datetime.now()
            logger.info(
                f"{operation}_started",
                operation=operation,
                **context
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

            if exc_type is None:
                logger.info(
                    f"{operation}_completed",
                    operation=operation,
                    duration_ms=round(duration_ms, 2),
                    status="success",
                    **context
                )
            else:
                logger.error(
                    f"{operation}_failed",
                    operation=operation,
                    duration_ms=round(duration_ms, 2),
                    status="error",
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    **context
                )
            return False  # Don't suppress exceptions

    return OperationLogger()


# Pre-configured loggers for different components
def get_api_logger() -> structlog.BoundLogger:
    """Logger for API endpoints."""
    return get_logger("callwhisper.api")


def get_service_logger() -> structlog.BoundLogger:
    """Logger for service layer."""
    return get_logger("callwhisper.services")


def get_core_logger() -> structlog.BoundLogger:
    """Logger for core components."""
    return get_logger("callwhisper.core")
