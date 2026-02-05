"""
Structured logging configuration for ML Platform.
"""

import logging
import sys
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    json_format: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for a service.

    Args:
        service_name: Name of the service for identification
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to use JSON formatting

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    logger.addHandler(handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name or "mlplatform")
