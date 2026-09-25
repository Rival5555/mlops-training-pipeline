"""Unit tests for configuration schema parsing and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pipeline.config.schema import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    load_config,
)


def test_valid_default_config() -> None:
    """Ensure default config instantiates without validation errors."""
    config = PipelineConfig()
    assert config.project_name == "automated-mlops-training-pipeline"
    assert config.data.train_ratio == 0.70
    assert config.data.val_ratio == 0.15
    assert config.data.test_ratio == 0.15


def test_split_ratios_must_sum_to_one() -> None:
    """Validate that split ratios not summing to 1.0 raise a clear ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        DataConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)  # Sums to 1.1

    assert "Split ratios must sum to 1.0" in str(exc_info.value)


def test_invalid_learning_rate() -> None:
    """Ensure non-positive learning rates are rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=-0.01)

    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=0.0)


def test_invalid_dropout_rate() -> None:
    """Ensure dropout rate is bounded between 0.0 and 0.8."""
    with pytest.raises(ValidationError):
        ModelConfig(dropout_rate=0.95)

    with pytest.raises(ValidationError):
        ModelConfig(dropout_rate=-0.1)


def test_config_hash_determinism() -> None:
    """Verify that identical configurations produce identical hashes."""
    cfg1 = PipelineConfig()
    cfg2 = PipelineConfig()
    assert cfg1.compute_hash() == cfg2.compute_hash()

    cfg3 = PipelineConfig(train=TrainingConfig(learning_rate=0.005))
    assert cfg1.compute_hash() != cfg3.compute_hash()


def test_load_config_reads_existing_file() -> None:
    """Test loader on repository config files."""
    config = load_config(Path("configs/config.yaml"))
    assert config.model.num_classes == 2
    assert config.evaluate.primary_metric == "f1_macro"
