"""Stage 2: Preprocessing, Deterministic Stratified Split, and Dataset Versioning.

Responsibilities:
- Ingest validated samples from Stage 1.
- Perform deterministic, stratified train/validation/test split using seeded random state.
- Create declarative manifest mapping file paths, class indices, and splits.
- Tag and record dataset version hash alongside configuration hash.
- Prepare processed dataset directory structure.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import click
from sklearn.model_selection import train_test_split

from src.pipeline.common.exceptions import PreprocessingError
from src.pipeline.common.logger import get_logger
from src.pipeline.config.schema import PipelineConfig, load_config

logger = get_logger("s02_preprocess_split")


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file for data version tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def run_preprocess_and_split(
    config: PipelineConfig | None = None, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Execute Preprocessing and Deterministic Split."""
    if config is None:
        config = load_config(config_path)

    raw_dir = config.data.raw_data_dir
    processed_dir = config.data.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    expected_classes = config.model.class_names
    class_to_idx = {name: idx for idx, name in enumerate(expected_classes)}
    seed = config.data.random_seed
    allowed_exts = tuple(ext.lower() for ext in config.data.allowed_extensions)

    logger.info("Starting dataset preprocessing and stratified splitting", seed=seed)

    # Collect samples
    all_filepaths: list[Path] = []
    all_labels: list[int] = []

    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = raw_dir / cls_name
        if not cls_dir.exists():
            raise PreprocessingError(f"Class folder not found: {cls_dir}")
        for img_path in sorted(cls_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in allowed_exts:
                all_filepaths.append(img_path)
                all_labels.append(cls_idx)

    if not all_filepaths:
        raise PreprocessingError(f"No valid image files found in {raw_dir}")

    total_samples = len(all_filepaths)
    val_ratio = config.data.val_ratio
    test_ratio = config.data.test_ratio

    # Calculate proportions
    val_test_ratio = val_ratio + test_ratio
    test_within_val_test = test_ratio / val_test_ratio

    # Step 1: Stratified split into Train vs (Val + Test)
    train_paths, val_test_paths, train_y, val_test_y = train_test_split(
        all_filepaths,
        all_labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=all_labels,
    )

    # Step 2: Stratified split (Val + Test) into Val and Test
    val_paths, test_paths, val_y, test_y = train_test_split(
        val_test_paths,
        val_test_y,
        test_size=test_within_val_test,
        random_state=seed,
        stratify=val_test_y,
    )

    logger.info(
        "Split completed",
        train_count=len(train_paths),
        val_count=len(val_paths),
        test_count=len(test_paths),
        total=total_samples,
    )

    # Build manifest entries with relative paths and hashes
    def build_records(
        paths: list[Path], labels: list[int], split_name: str
    ) -> list[dict[str, Any]]:
        records = []
        for p, y in zip(paths, labels, strict=False):
            rel_path = p.relative_to(raw_dir.parent) if p.is_relative_to(raw_dir.parent) else p
            records.append(
                {
                    "filepath": rel_path.as_posix(),
                    "label": y,
                    "class_name": expected_classes[y],
                    "split": split_name,
                    "sha256": compute_file_sha256(p),
                }
            )
        return records

    train_records = build_records(train_paths, train_y, "train")
    val_records = build_records(val_paths, val_y, "val")
    test_records = build_records(test_paths, test_y, "test")

    # Compute overall dataset fingerprint
    all_sha = sorted(r["sha256"] for r in (train_records + val_records + test_records))
    dataset_fingerprint = hashlib.sha256("".join(all_sha).encode("utf-8")).hexdigest()[:16]
    config_hash = config.compute_hash()

    manifest = {
        "dataset_version_hash": dataset_fingerprint,
        "config_hash": config_hash,
        "seed": seed,
        "classes": expected_classes,
        "class_to_idx": class_to_idx,
        "counts": {
            "total": total_samples,
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "records": {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        },
    }

    manifest_path = processed_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Saved split manifest",
        manifest_path=str(manifest_path),
        dataset_version_hash=dataset_fingerprint,
        config_hash=config_hash,
    )

    return manifest


@click.command()
@click.option(
    "--config", "config_path", default="configs/config.yaml", help="Path to master config YAML"
)
def main(config_path: str) -> None:
    """CLI entrypoint for Stage 2: Preprocessing and Split."""
    try:
        run_preprocess_and_split(config_path=config_path)
    except PreprocessingError as e:
        logger.error("Preprocessing stage failed!", error=str(e))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
