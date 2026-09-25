"""Typed domain exceptions for the MLOps training pipeline.

Ensures explicit, loud failure modes instead of silent degradations or generic exceptions.
"""

from typing import Any


class PipelineBaseException(Exception):
    """Base exception for all pipeline-related failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigValidationError(PipelineBaseException):
    """Raised when pipeline configuration is invalid or missing required keys."""

    pass


class DataValidationError(PipelineBaseException):
    """Raised when data ingestion, schema, image integrity, or class balance checks fail."""

    pass


class PreprocessingError(PipelineBaseException):
    """Raised when deterministic splitting, transformations, or dataset manifests fail."""

    pass


class TrainingError(PipelineBaseException):
    """Raised when training loop encounters unrecoverable errors (e.g., NaN loss, divergence)."""

    pass


class EvaluationGateError(PipelineBaseException):
    """Raised when candidate model fails quality gate criteria (e.g. does not beat baseline)."""

    pass


class ModelRegistrationError(PipelineBaseException):
    """Raised when MLflow model registry operations fail."""

    pass


class ParityCheckError(PipelineBaseException):
    """Raised when exported ONNX model outputs do not match PyTorch outputs within tolerance."""

    pass
