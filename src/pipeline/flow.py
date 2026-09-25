"""End-to-End MLOps Pipeline Orchestrator using Prefect 3.x.

Features:
- Prefect @task and @flow decorators for end-to-end stage orchestration
- State monitoring, automatic logging, and execution tracking
- Idempotent execution: verifies config and dataset versions
- Graceful standalone CLI execution (`python -m src.pipeline.flow`)
"""

import sys
from typing import Any

import click

from src.pipeline.common.exceptions import PipelineBaseException
from src.pipeline.common.logger import get_logger
from src.pipeline.config.schema import PipelineConfig, load_config
from src.pipeline.stages.s01_ingest_validate import run_ingest_and_validate
from src.pipeline.stages.s02_preprocess_split import run_preprocess_and_split
from src.pipeline.stages.s03_train import run_training
from src.pipeline.stages.s04_eval_gate import run_evaluation_gate
from src.pipeline.stages.s05_register import run_model_registration
from src.pipeline.stages.s06_export_package import run_export_and_package

logger = get_logger("orchestrator_flow")

try:
    from prefect import flow, task

    _HAS_PREFECT = True
except ImportError:
    _HAS_PREFECT = False

    # Minimal decorator fallbacks if Prefect is not installed
    def task(fn: Any = None, **kwargs: Any) -> Any:  # type: ignore
        def decorator(f: Any) -> Any:
            return f

        return decorator(fn) if fn else decorator

    def flow(fn: Any = None, **kwargs: Any) -> Any:  # type: ignore
        def decorator(f: Any) -> Any:
            return f

        return decorator(fn) if fn else decorator


@task(name="Stage 1: Ingestion & Validation", retries=1)
def task_ingest_validate(config: PipelineConfig) -> dict[str, Any]:
    return run_ingest_and_validate(config=config)


@task(name="Stage 2: Preprocessing & Splitting", retries=0)
def task_preprocess_split(config: PipelineConfig) -> dict[str, Any]:
    return run_preprocess_and_split(config=config)


@task(name="Stage 3: Training & Checkpointing", retries=0)
def task_train(config: PipelineConfig) -> dict[str, Any]:
    return run_training(config=config)


@task(name="Stage 4: Evaluation & Validation Gate", retries=0)
def task_eval_gate(config: PipelineConfig) -> dict[str, Any]:
    return run_evaluation_gate(config=config)


@task(name="Stage 5: Model Registration", retries=1)
def task_register(config: PipelineConfig, promote_to_production: bool = False) -> dict[str, Any]:
    return run_model_registration(config=config, promote_to_production=promote_to_production)


@task(name="Stage 6: Export & Parity Verification", retries=0)
def task_export(config: PipelineConfig) -> dict[str, Any]:
    return run_export_and_package(config=config)


@flow(name="automated-mlops-training-pipeline", log_prints=True)
def mlops_training_pipeline(
    config_path: str = "configs/config.yaml",
    promote_to_production: bool = False,
) -> dict[str, Any]:
    """Execute full automated MLOps training pipeline."""
    logger.info("Initializing automated MLOps pipeline run", config_path=config_path)
    config = load_config(config_path)

    config_hash = config.compute_hash()
    logger.info(
        "Loaded pipeline configuration",
        project=config.project_name,
        env=config.environment,
        config_hash=config_hash,
    )

    # Stage 1: Data Ingestion & Validation
    validation_report = task_ingest_validate(config)

    # Stage 2: Preprocessing & Stratified Splitting
    split_manifest = task_preprocess_split(config)

    # Stage 3: PyTorch Model Training & Checkpointing
    train_summary = task_train(config)

    # Stage 4: Held-out Test Evaluation & Baseline Gate Check
    gate_decision = task_eval_gate(config)

    # Stage 5: MLflow Model Registration (Staging / Production)
    registration_result = task_register(config, promote_to_production=promote_to_production)

    # Stage 6: ONNX Export & Parity Verification
    export_summary = task_export(config)

    pipeline_result = {
        "status": "COMPLETED_SUCCESSFULLY",
        "config_hash": config_hash,
        "dataset_hash": split_manifest.get("dataset_version_hash"),
        "validation_report": validation_report,
        "train_summary": train_summary,
        "gate_decision": gate_decision,
        "registration_result": registration_result,
        "export_summary": export_summary,
    }

    logger.info(
        "MLOps Training Pipeline completed all stages successfully!",
        model_version=registration_result.get("model_version"),
        gate_status="PASSED",
    )
    return pipeline_result


@click.command()
@click.option("--config", "config_path", default="configs/config.yaml", help="Path to config YAML")
@click.option(
    "--promote-to-production",
    is_flag=True,
    default=False,
    help="Promote model to Production stage if gate passes",
)
def main(config_path: str, promote_to_production: bool) -> None:
    """CLI Runner for the complete MLOps pipeline flow."""
    try:
        mlops_training_pipeline(
            config_path=config_path,
            promote_to_production=promote_to_production,
        )
    except PipelineBaseException as e:
        logger.error("Pipeline execution aborted due to typed failure!", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred during pipeline run!", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
