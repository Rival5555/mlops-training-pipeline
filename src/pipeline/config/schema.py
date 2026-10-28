"""Strongly-typed Pydantic configuration schemas and validation for the MLOps pipeline."""

import hashlib
import os
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, model_validator


class GateStrategy(str, Enum):
    FAIL = "fail"
    WARN = "warn"


class NormalizationConfig(BaseModel):
    mean: list[float] = Field(default=[0.485, 0.456, 0.406])
    std: list[float] = Field(default=[0.229, 0.224, 0.225])

    @model_validator(mode="after")
    def validate_channels(self) -> "NormalizationConfig":
        if len(self.mean) != len(self.std):
            raise ValueError("Normalization mean and std must have the same number of channels")
        if not self.mean:
            raise ValueError("Normalization mean and std must not be empty")
        return self


class AugmentationConfig(BaseModel):
    horizontal_flip_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_degrees: int = Field(default=15, ge=0, le=180)
    color_jitter_brightness: float = Field(default=0.1, ge=0.0, le=1.0)
    color_jitter_contrast: float = Field(default=0.1, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)
    image_size: tuple[int, int] = Field(default=(224, 224))
    num_channels: int = Field(default=3, ge=1, le=4)
    allowed_extensions: list[str] = Field(default=[".jpg", ".jpeg", ".png"])
    min_samples_per_class: int = Field(default=10, ge=2)
    class_imbalance_max_ratio: float = Field(default=10.0, gt=1.0)
    batch_size: int = Field(default=16, ge=1)
    num_workers: int = Field(default=0, ge=0)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DataConfig":
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 4)
        if total != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0! Got train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio} (sum={total})"
            )
        return self


class ModelConfig(BaseModel):
    architecture: str = Field(default="efficientnet_b0")
    num_classes: int = Field(default=2, ge=2)
    pretrained: bool = Field(default=True)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.8)
    freeze_backbone: bool = Field(default=False)
    class_names: list[str] = Field(default=["healthy", "referable_dr"])

    @model_validator(mode="after")
    def validate_class_names(self) -> "ModelConfig":
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"num_classes ({self.num_classes}) must match class_names length "
                f"({len(self.class_names)})"
            )
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if any(not name.strip() for name in self.class_names):
            raise ValueError("class_names must not contain blank names")
        return self


class TrainingConfig(BaseModel):
    SUPPORTED_SCHEDULERS: ClassVar[set[str]] = {"cosine", "none"}
    SUPPORTED_EARLY_STOPPING_METRICS: ClassVar[set[str]] = {"val_loss", "val_f1"}
    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.0003, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    optimizer: str = Field(default="adamw")
    scheduler: str = Field(default="cosine")
    mixed_precision: bool = Field(default=True)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_metric: str = Field(default="val_loss")
    early_stopping_mode: str = Field(default="min")
    checkpoint_dir: Path = Field(default=Path("models/checkpoints"))
    save_best_and_last: bool = Field(default=True)
    log_interval_steps: int = Field(default=10, ge=1)
    seed: int = Field(default=42)

    @model_validator(mode="after")
    def validate_training_options(self) -> "TrainingConfig":
        if self.scheduler.lower() not in self.SUPPORTED_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'. "
                f"Choose one of {sorted(self.SUPPORTED_SCHEDULERS)}"
            )
        if self.early_stopping_metric.lower() not in self.SUPPORTED_EARLY_STOPPING_METRICS:
            raise ValueError(
                f"Unsupported early_stopping_metric '{self.early_stopping_metric}'. "
                f"Choose one of {sorted(self.SUPPORTED_EARLY_STOPPING_METRICS)}"
            )
        if self.early_stopping_mode.lower() not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be either 'min' or 'max'")
        if self.early_stopping_metric.lower() == "val_loss" and self.early_stopping_mode.lower() != "min":
            raise ValueError("val_loss must use early_stopping_mode='min'")
        if self.early_stopping_metric.lower() == "val_f1" and self.early_stopping_mode.lower() != "max":
            raise ValueError("val_f1 must use early_stopping_mode='max'")
        return self


class EvaluationGateConfig(BaseModel):
    SUPPORTED_METRICS: ClassVar[set[str]] = {
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
    }
    primary_metric: str = Field(default="f1_macro")
    min_absolute_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    min_improvement_over_production: float = Field(default=0.01, ge=0.0)
    allow_first_model: bool = Field(default=True)
    metrics_to_compute: list[str] = Field(
        default=["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    )
    gate_decision_output: Path = Field(default=Path("models/checkpoints/gate_decision.json"))
    gate_strategy: GateStrategy = Field(default=GateStrategy.FAIL)

    @model_validator(mode="after")
    def validate_metrics(self) -> "EvaluationGateConfig":
        if self.primary_metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported primary_metric '{self.primary_metric}'. "
                f"Choose one of {sorted(self.SUPPORTED_METRICS)}"
            )
        unsupported = set(self.metrics_to_compute) - self.SUPPORTED_METRICS
        if unsupported:
            raise ValueError(f"Unsupported metrics_to_compute: {sorted(unsupported)}")
        return self


class ExportConfig(BaseModel):
    export_format: str = Field(default="onnx")
    opset_version: int = Field(default=17, ge=11, le=19)
    dynamic_batch: bool = Field(default=True)
    output_dir: Path = Field(default=Path("models/exported"))
    output_filename: str = Field(default="model.onnx")
    float_tolerance_atol: float = Field(default=0.0001, gt=0.0)
    float_tolerance_rtol: float = Field(default=0.001, gt=0.0)
    dummy_input_shape: list[int] = Field(default=[1, 3, 224, 224])
    verify_onnx_runtime: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_export_format(self) -> "ExportConfig":
        if self.export_format.lower() != "onnx":
            raise ValueError("Only 'onnx' export_format is supported")
        return self


class MLflowConfig(BaseModel):
    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="retinopathy_classifier_experiment")
    model_name: str = Field(default="retinopathy_classifier")
    staging_tag: str = Field(default="staging")
    production_tag: str = Field(default="production")
    auto_register: bool = Field(default=True)


class PipelineConfig(BaseModel):
    project_name: str = Field(default="automated-mlops-training-pipeline")
    environment: str = Field(default="development")
    pipeline_version: str = Field(default="1.0.0")
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluate: EvaluationGateConfig = Field(default_factory=EvaluationGateConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    def compute_hash(self) -> str:
        """Compute deterministic SHA256 hash of the full configuration."""
        dumped = self.model_dump_json()
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def _read_yaml(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}
    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load, merge, and validate pipeline configuration from YAML and env vars."""
    base_path = Path(config_path) if config_path else Path("configs/config.yaml")
    config_dir = base_path.parent if base_path.exists() else Path("configs")

    # Load master config
    raw_master = _read_yaml(base_path) if base_path.exists() else {}

    # Load sub-configs if present
    data_dict = _read_yaml(config_dir / "data" / "default.yaml")
    model_dict = _read_yaml(config_dir / "model" / "efficientnet_b0.yaml")
    train_dict = _read_yaml(config_dir / "train" / "default.yaml")
    eval_dict = _read_yaml(config_dir / "evaluate" / "gate.yaml")
    export_dict = _read_yaml(config_dir / "export" / "onnx.yaml")

    # Merge nested dictionaries
    merged_data = {**data_dict, **raw_master.get("data", {})}
    merged_model = {**model_dict, **raw_master.get("model", {})}
    merged_train = {**train_dict, **raw_master.get("train", {})}
    merged_eval = {**eval_dict, **raw_master.get("evaluate", {})}
    merged_export = {**export_dict, **raw_master.get("export", {})}
    merged_mlflow = {**raw_master.get("mlflow", {})}

    # Read environment variable overrides
    if env_mlflow_uri := os.getenv("MLFLOW_TRACKING_URI"):
        merged_mlflow["tracking_uri"] = env_mlflow_uri
    if env_experiment_name := os.getenv("MLFLOW_EXPERIMENT_NAME"):
        merged_mlflow["experiment_name"] = env_experiment_name
    if env_model_name := os.getenv("MLFLOW_MODEL_NAME"):
        merged_mlflow["model_name"] = env_model_name

    final_dict = {
        "project_name": raw_master.get("project_name", "automated-mlops-training-pipeline"),
        "environment": raw_master.get("environment", "development"),
        "pipeline_version": raw_master.get("pipeline_version", "1.0.0"),
        "data": merged_data,
        "model": merged_model,
        "train": merged_train,
        "evaluate": merged_eval,
        "export": merged_export,
        "mlflow": merged_mlflow,
    }

    return PipelineConfig.model_validate(final_dict)
