"""Integration smoke test running all pipeline stages sequentially on synthetic micro-dataset."""

from pathlib import Path

import pytest

from src.pipeline.config.schema import PipelineConfig
from src.pipeline.flow import run_pipeline_e2e
from src.pipeline.stages.s01_ingest_validate import run_ingest_and_validate
from src.pipeline.stages.s02_preprocess_split import run_preprocess_and_split
from src.pipeline.stages.s03_train import run_training
from src.pipeline.stages.s04_eval_gate import run_evaluation_gate
from src.pipeline.stages.s05_register import run_model_registration
from src.pipeline.stages.s06_export_package import run_export_and_package


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_pipeline_end_to_end_smoke(
    base_test_config: PipelineConfig, synthetic_raw_dataset: Path
) -> None:
    """Execute stages 1 through 6 sequentially, ensuring artifact continuity and gate compliance."""
    config = base_test_config
    config.data.raw_data_dir = synthetic_raw_dataset
    config.evaluate.min_absolute_threshold = 0.20

    # Stage 1: Ingest & Validate
    val_report = run_ingest_and_validate(config=config)
    assert val_report["status"] == "VALIDATED"

    # Stage 2: Preprocess & Split
    manifest = run_preprocess_and_split(config=config)
    assert (config.data.processed_data_dir / "split_manifest.json").exists()
    assert manifest["counts"]["total"] == 30

    # Stage 3: Train & Checkpoint
    train_summary = run_training(config=config)
    assert (config.train.checkpoint_dir / "best_model.pt").exists()
    assert train_summary["run_id"] is not None

    # Stage 4: Held-out Test Evaluation & Baseline Gate
    gate_decision = run_evaluation_gate(config=config)
    assert gate_decision["passed"] is True
    assert (config.evaluate.gate_decision_output).exists()

    # Stage 5: Model Registration
    reg_result = run_model_registration(config=config, promote_to_production=False)
    assert reg_result["target_stage"] == "Staging"
    assert (config.train.checkpoint_dir / "registration_receipt.json").exists()

    # Stage 6: Export & Smoke Parity Test
    export_summary = run_export_and_package(config=config)
    assert export_summary["parity_verified"] is True
    assert (config.export.output_dir / config.export.output_filename).exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_pipeline_run_pipeline_e2e(
    base_test_config: PipelineConfig, synthetic_raw_dataset: Path
) -> None:
    """Ensure run_pipeline_e2e executes all 6 stages sequentially and returns valid result."""
    config = base_test_config
    config.data.raw_data_dir = synthetic_raw_dataset
    config.evaluate.min_absolute_threshold = 0.20

    result = run_pipeline_e2e(config=config, promote_to_production=False)
    assert result["status"] == "COMPLETED_SUCCESSFULLY"
    assert result["validation_report"]["status"] == "VALIDATED"
    assert result["gate_decision"]["passed"] is True
    assert result["registration_result"]["target_stage"] == "Staging"
    assert result["export_summary"]["parity_verified"] is True
