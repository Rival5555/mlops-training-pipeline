"""Structured logging configuration for pipeline stages."""

import logging
import sys
from typing import Any

try:
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


class PipelineLogger:
    """Wrapper that provides structured logging with contextual metadata."""

    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        if _HAS_STRUCTLOG:
            self._logger = structlog.get_logger(stage=stage_name)
        else:
            self._logger = logging.getLogger(stage_name)
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)

    def info(self, event: str, **kwargs: Any) -> None:
        if _HAS_STRUCTLOG:
            self._logger.info(event, **kwargs)
        else:
            extra_str = f" | {kwargs}" if kwargs else ""
            self._logger.info(f"{event}{extra_str}")

    def warning(self, event: str, **kwargs: Any) -> None:
        if _HAS_STRUCTLOG:
            self._logger.warning(event, **kwargs)
        else:
            extra_str = f" | {kwargs}" if kwargs else ""
            self._logger.warning(f"{event}{extra_str}")

    def debug(self, event: str, **kwargs: Any) -> None:
        if _HAS_STRUCTLOG:
            self._logger.debug(event, **kwargs)
        else:
            extra_str = f" | {kwargs}" if kwargs else ""
            self._logger.debug(f"{event}{extra_str}")

    def error(self, event: str, **kwargs: Any) -> None:
        if _HAS_STRUCTLOG:
            self._logger.error(event, **kwargs)
        else:
            extra_str = f" | {kwargs}" if kwargs else ""
            self._logger.error(f"{event}{extra_str}")

    def exception(self, event: str, **kwargs: Any) -> None:
        if _HAS_STRUCTLOG:
            self._logger.exception(event, **kwargs)
        else:
            extra_str = f" | {kwargs}" if kwargs else ""
            self._logger.exception(f"{event}{extra_str}")


def get_logger(stage_name: str) -> PipelineLogger:
    """Factory function to acquire a stage logger."""
    return PipelineLogger(stage_name=stage_name)
