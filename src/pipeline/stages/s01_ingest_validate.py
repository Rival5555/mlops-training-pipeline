"""Stage 1: Data Ingestion and Validation.

Checks:
- Existence of data directory and class subdirectories / labels
- Schema validation against configured classes
- File corruption check (PIL header verification and image decoding)
- Minimum sample threshold per class
- Severe class imbalance threshold check
- Fails loudly with DataValidationError upon any violation
"""

import json
from pathlib import Path
from typing import Any

import click
from PIL import Image

from src.pipeline.common.exceptions import DataValidationError
from src.pipeline.common.logger import get_logger
from src.pipeline.config.schema import PipelineConfig, load_config

logger = get_logger("s01_ingest_validate")


def validate_image_file(file_path: Path, min_size: tuple[int, int] = (16, 16)) -> None:
    """Verify that the file is an intact, readable image.

    Raises DataValidationError if corrupted or invalid.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Verify decoding pixel data
        with Image.open(file_path) as img:
            img.load()
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                raise DataValidationError(
                    f"Image {file_path.name} dimension {img.size} is smaller than minimum required {min_size}",
                    {"file": str(file_path), "dimensions": img.size},
                )
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(
            f"Corrupted or unreadable image file: {file_path.name}",
            {"file": str(file_path), "error": str(e)},
        ) from e


def run_ingest_and_validate(
    config: PipelineConfig | None = None, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Execute Data Ingestion & Validation stage."""
    if config is None:
        config = load_config(config_path)

    raw_dir = config.data.raw_data_dir
    expected_classes = config.model.class_names
    min_samples = config.data.min_samples_per_class
    max_imbalance_ratio = config.data.class_imbalance_max_ratio
    allowed_exts = tuple(ext.lower() for ext in config.data.allowed_extensions)

    logger.info(
        "Starting data ingestion and quality validation",
        raw_data_dir=str(raw_dir),
        expected_classes=expected_classes,
    )

    if not raw_dir.exists():
        raise DataValidationError(
            f"Raw data directory does not exist: {raw_dir}",
            {"raw_data_dir": str(raw_dir)},
        )

    # Inspect class directories
    class_counts: dict[str, int] = {}
    verified_files: list[dict[str, str]] = []
    corrupted_files: list[dict[str, str]] = []

    for cls_name in expected_classes:
        cls_dir = raw_dir / cls_name
        if not cls_dir.exists() or not cls_dir.is_dir():
            raise DataValidationError(
                f"Missing expected class directory: '{cls_name}' under {raw_dir}",
                {"missing_class": cls_name, "raw_dir": str(raw_dir)},
            )

        files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in allowed_exts]

        count = 0
        for f in files:
            try:
                validate_image_file(f)
                verified_files.append({"path": str(f), "class": cls_name})
                count += 1
            except DataValidationError as err:
                corrupted_files.append({"path": str(f), "reason": err.message})

        class_counts[cls_name] = count

    # Fail loudly if any images were corrupted
    if corrupted_files:
        raise DataValidationError(
            f"Validation failed: detected {len(corrupted_files)} corrupted image files!",
            {"corrupted_count": len(corrupted_files), "corrupted_samples": corrupted_files[:5]},
        )

    # Check minimum samples per class
    for cls_name, count in class_counts.items():
        if count < min_samples:
            raise DataValidationError(
                f"Class '{cls_name}' has {count} samples, fewer than minimum threshold ({min_samples})",
                {"class": cls_name, "count": count, "min_required": min_samples},
            )

    # Check class balance ratio
    counts_list = list(class_counts.values())
    min_count = min(counts_list)
    max_count = max(counts_list)
    imbalance_ratio = max_count / max(min_count, 1)

    if imbalance_ratio > max_imbalance_ratio:
        raise DataValidationError(
            f"Severe class imbalance detected: ratio {imbalance_ratio:.2f} exceeds max {max_imbalance_ratio}",
            {
                "imbalance_ratio": imbalance_ratio,
                "max_allowed": max_imbalance_ratio,
                "class_counts": class_counts,
            },
        )

    total_samples = sum(counts_list)
    report = {
        "status": "VALIDATED",
        "total_samples": total_samples,
        "class_counts": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "classes": expected_classes,
        "raw_dir": str(raw_dir),
    }

    # Save validation report
    report_path = raw_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    logger.info(
        "Data validation successfully passed",
        total_samples=total_samples,
        class_counts=class_counts,
        imbalance_ratio=round(imbalance_ratio, 2),
        report_path=str(report_path),
    )

    return report


@click.command()
@click.option(
    "--config", "config_path", default="configs/config.yaml", help="Path to master config YAML"
)
def main(config_path: str) -> None:
    """CLI entrypoint for Stage 1: Data Ingestion and Validation."""
    try:
        run_ingest_and_validate(config_path=config_path)
    except DataValidationError as e:
        logger.error("Data validation stage failed loudly!", error=str(e))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
