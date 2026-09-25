"""Common utilities, logging, exceptions, and MLflow helpers."""

from src.pipeline.common.exceptions import (
    ConfigValidationError,
    DataValidationError,
    EvaluationGateError,
    ModelRegistrationError,
    ParityCheckError,
    PipelineBaseException,
    PreprocessingError,
    TrainingError,
)
from src.pipeline.common.logger import get_logger
from src.pipeline.common.mlflow_utils import (
    get_git_commit_hash,
    get_production_model_metrics,
    log_confusion_matrix,
    setup_mlflow,
)

__all__ = [
    "PipelineBaseException",
    "ConfigValidationError",
    "DataValidationError",
    "PreprocessingError",
    "TrainingError",
    "EvaluationGateError",
    "ModelRegistrationError",
    "ParityCheckError",
    "get_logger",
    "setup_mlflow",
    "get_production_model_metrics",
    "log_confusion_matrix",
    "get_git_commit_hash",
]
