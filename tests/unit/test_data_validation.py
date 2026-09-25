"""Unit tests for Stage 1: Data Ingestion and Validation logic."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipeline.common.exceptions import DataValidationError
from src.pipeline.config.schema import PipelineConfig
from src.pipeline.stages.s01_ingest_validate import run_ingest_and_validate, validate_image_file


def test_validate_image_file_valid(sample_image: Path) -> None:
    """Valid image passes integrity check without error."""
    validate_image_file(sample_image)


def test_validate_image_file_corrupted(corrupt_image: Path) -> None:
    """Corrupted image raises DataValidationError."""
    with pytest.raises(DataValidationError) as exc:
        validate_image_file(corrupt_image)
    assert "Corrupted or unreadable image" in str(exc.value)


def test_validate_image_too_small(tmp_path: Path) -> None:
    """Image below minimum dimensions raises DataValidationError."""
    small_img = tmp_path / "tiny.jpg"
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr).save(small_img, "JPEG")

    with pytest.raises(DataValidationError) as exc:
        validate_image_file(small_img, min_size=(16, 16))
    assert "smaller than minimum required" in str(exc.value)


def test_run_ingest_and_validate_success(
    base_test_config: PipelineConfig, synthetic_raw_dataset: Path
) -> None:
    """End-to-end stage validation passes on healthy dataset."""
    base_test_config.data.raw_data_dir = synthetic_raw_dataset
    report = run_ingest_and_validate(config=base_test_config)

    assert report["status"] == "VALIDATED"
    assert report["total_samples"] == 30
    assert report["class_counts"] == {"healthy": 15, "referable_dr": 15}
    assert (synthetic_raw_dataset / "validation_report.json").exists()


def test_run_ingest_missing_raw_dir(base_test_config: PipelineConfig, tmp_path: Path) -> None:
    """Non-existent raw data directory raises DataValidationError."""
    base_test_config.data.raw_data_dir = tmp_path / "non_existent_dir"
    with pytest.raises(DataValidationError) as exc:
        run_ingest_and_validate(config=base_test_config)
    assert "Raw data directory does not exist" in str(exc.value)


def test_run_ingest_missing_class_folder(base_test_config: PipelineConfig, tmp_path: Path) -> None:
    """Missing class folder raises DataValidationError."""
    raw_dir = tmp_path / "raw_partial"
    raw_dir.mkdir(parents=True)
    # Only create "healthy", omit "referable_dr"
    (raw_dir / "healthy").mkdir()
    base_test_config.data.raw_data_dir = raw_dir

    with pytest.raises(DataValidationError) as exc:
        run_ingest_and_validate(config=base_test_config)
    assert "Missing expected class directory" in str(exc.value)


def test_run_ingest_detects_corrupt_file_in_dataset(
    base_test_config: PipelineConfig, synthetic_raw_dataset: Path
) -> None:
    """A single corrupted file inside a class folder causes stage to fail loudly."""
    base_test_config.data.raw_data_dir = synthetic_raw_dataset
    bad_file = synthetic_raw_dataset / "healthy" / "bad_image.jpg"
    with open(bad_file, "wb") as f:
        f.write(b"this is corrupt junk data")

    with pytest.raises(DataValidationError) as exc:
        run_ingest_and_validate(config=base_test_config)
    assert "corrupted image files" in str(exc.value)


def test_run_ingest_insufficient_samples(base_test_config: PipelineConfig, tmp_path: Path) -> None:
    """Having fewer than min_samples_per_class raises DataValidationError."""
    raw_dir = tmp_path / "raw_scant"
    for cls in ["healthy", "referable_dr"]:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True)
        # Only 2 samples when min required is 5
        for i in range(2):
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg", "JPEG")

    base_test_config.data.raw_data_dir = raw_dir
    base_test_config.data.min_samples_per_class = 5

    with pytest.raises(DataValidationError) as exc:
        run_ingest_and_validate(config=base_test_config)
    assert "fewer than minimum threshold" in str(exc.value)


def test_run_ingest_severe_class_imbalance(
    base_test_config: PipelineConfig, tmp_path: Path
) -> None:
    """Class imbalance ratio exceeding threshold raises DataValidationError."""
    raw_dir = tmp_path / "raw_imbalanced"
    # healthy: 50, referable_dr: 3 -> ratio 16.66 > max allowed 5.0
    for cls, count in [("healthy", 50), ("referable_dr", 3)]:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(count):
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg", "JPEG")

    base_test_config.data.raw_data_dir = raw_dir
    base_test_config.data.min_samples_per_class = 2
    base_test_config.data.class_imbalance_max_ratio = 5.0

    with pytest.raises(DataValidationError) as exc:
        run_ingest_and_validate(config=base_test_config)
    assert "Severe class imbalance detected" in str(exc.value)
