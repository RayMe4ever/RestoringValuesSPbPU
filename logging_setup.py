# logging_setup.py
from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

DEFAULT_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "pid=%(process)d op_id=%(op_id)s %(message)s"
)


class UtcIsoFormatter(logging.Formatter):
    """UTC timestamps in ISO-8601 with milliseconds."""

    def formatTime(self, record: logging.LogRecord, datefmt=None) -> str:  # noqa: D401
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class OpIdFilter(logging.Filter):
    """Ensures every record has op_id field."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "op_id"):
            record.op_id = "-"
        return True


def new_op_id(prefix: str | None = None) -> str:
    base = uuid.uuid4().hex
    return f"{prefix}-{base}" if prefix else base


def setup_logging(component: str) -> logging.Logger:
    """
    Configure root logging once per process.

    Env:
      LOG_LEVEL: INFO/DEBUG/WARNING/ERROR (default INFO)
      LOG_FORMAT: override DEFAULT_FORMAT
      LOG_FILE: optional file path, e.g. logs/app.log
    """
    root = logging.getLogger()
    # Guard against duplicate configuration (Dash reloader / repeated imports)
    if getattr(root, "_rvspu_configured", False):
        return logging.getLogger(component)

    level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    fmt = os.getenv("LOG_FORMAT") or DEFAULT_FORMAT
    log_file = os.getenv("LOG_FILE")

    root.setLevel(level)

    formatter = UtcIsoFormatter(fmt)
    op_filter = OpIdFilter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(op_filter)
    root.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(op_filter)
        root.addHandler(file_handler)

    root._rvspu_configured = True  # type: ignore[attr-defined]
    logger = logging.getLogger(component)
    logger.debug("logging configured", extra={"op_id": new_op_id("log")})
    return logger
