"""Pytest fixtures for automated MLOps training pipeline tests."""

import shutil
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipeline.config.schema import (
    DataConfig,
    EvaluationGateConfig,
    ExportConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Temporary workspace directory for tests."""
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Generate a valid sample image."""
    img_path = tmp_path / "valid.jpg"
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path, "JPEG")
    return img_path


@pytest.fixture
def corrupt_image(tmp_path: Path) -> Path:
    """Generate a corrupt image file with broken header/bytes."""
    img_path = tmp_path / "corrupt.jpg"
    with open(img_path, "wb") as f:
        f.write(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"randomjunkdata" * 10
        )
    return img_path


@pytest.fixture
def synthetic_raw_dataset(tmp_path: Path) -> Path:
    """Create a mini valid dataset with 2 classes (15 samples each)."""
    raw_dir = tmp_path / "data" / "raw"
    classes = ["healthy", "referable_dr"]
    for cls in classes:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(15):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i:03d}.jpg", "JPEG")
    return raw_dir


@pytest.fixture
def base_test_config(tmp_path: Path) -> PipelineConfig:
    """Provide a fast, lightweight PipelineConfig configured for local testing."""
    return PipelineConfig(
        project_name="test-pipeline",
        environment="test",
        data=DataConfig(
            raw_data_dir=tmp_path / "data" / "raw",
            processed_data_dir=tmp_path / "data" / "processed",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            image_size=(64, 64),
            min_samples_per_class=5,
            class_imbalance_max_ratio=5.0,
            batch_size=4,
            num_workers=0,
        ),
        model=ModelConfig(
            architecture="resnet18",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.1,
            class_names=["healthy", "referable_dr"],
        ),
        train=TrainingConfig(
            epochs=1,
            learning_rate=0.001,
            mixed_precision=False,
            checkpoint_dir=tmp_path / "models" / "checkpoints",
            early_stopping_patience=2,
        ),
        evaluate=EvaluationGateConfig(
            primary_metric="f1_macro",
            min_absolute_threshold=0.50,
            min_improvement_over_production=0.01,
            allow_first_model=True,
            gate_decision_output=tmp_path / "models" / "checkpoints" / "gate_decision.json",
        ),
        export=ExportConfig(
            output_dir=tmp_path / "models" / "exported",
            dummy_input_shape=[1, 3, 64, 64],
            verify_onnx_runtime=True,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"sqlite:///{tmp_path.as_posix()}/mlflow.db",
            experiment_name="test_experiment",
            model_name="test_model",
        ),
    )
