"""Stage 5: Model Registration and Lifecycle Management in MLflow.

Responsibilities:
- Verify that Stage 4 evaluation gate passed.
- Register candidate model checkpoint into the MLflow Model Registry.
- Attach complete lineage metadata tags (git commit, dataset hash, config hash, test metrics).
- Default stage: 'staging'.
- Provides explicit promotion mechanism to 'production' via approval flag or workflow step.
"""

import json
from pathlib import Path
from typing import Any

import click

from src.pipeline.common.exceptions import ModelRegistrationError
from src.pipeline.common.logger import get_logger
from src.pipeline.common.mlflow_utils import get_git_commit_hash, setup_mlflow
from src.pipeline.config.schema import PipelineConfig, load_config

logger = get_logger("s05_register")


def run_model_registration(
    config: PipelineConfig | None = None,
    config_path: str | Path | None = None,
    promote_to_production: bool = False,
) -> dict[str, Any]:
    """Register validated model into MLflow Model Registry."""
    if config is None:
        config = load_config(config_path)

    # Verify Stage 4 gate decision
    gate_decision_file = config.evaluate.gate_decision_output
    if not gate_decision_file.exists():
        raise ModelRegistrationError(
            f"Gate decision file not found at {gate_decision_file}. Run Stage 4 first."
        )

    with open(gate_decision_file, encoding="utf-8") as f:
        gate_decision = json.load(f)

    if not gate_decision.get("passed", False):
        raise ModelRegistrationError(
            "Cannot register model: Candidate model FAILED the evaluation validation gate!",
            {"gate_decision": gate_decision},
        )

    run_id = gate_decision.get("run_id")
    if not run_id:
        # Check training summary
        train_summary_file = config.train.checkpoint_dir / "training_summary.json"
        if train_summary_file.exists():
            with open(train_summary_file, encoding="utf-8") as f:
                run_id = json.load(f).get("run_id")

    if not run_id:
        raise ModelRegistrationError("Run ID not found for registration candidate.")

    client = setup_mlflow(config.mlflow)
    model_name = config.mlflow.model_name
    model_uri = f"runs:/{run_id}/checkpoints"

    logger.info("Registering model in MLflow Registry", model_name=model_name, run_id=run_id)

    try:
        # Create registered model if it does not already exist
        try:
            client.get_registered_model(model_name)
        except Exception:
            logger.info(
                "Registered model does not exist yet. Creating registered model entity",
                model_name=model_name,
            )
            client.create_registered_model(
                name=model_name,
                description="Production Automated Retinopathy Image Classification Model",
            )

        # Register model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description="Automated CI/CD candidate model passed validation gate",
        )

        version_id = str(model_version.version)
        logger.info("Created model version", version=version_id, model_name=model_name)

        # Add Lineage and Metric Metadata Tags
        manifest_path = config.data.processed_data_dir / "split_manifest.json"
        data_hash = "unknown"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                data_hash = json.load(f).get("dataset_version_hash", "unknown")

        git_commit = get_git_commit_hash()
        config_hash = config.compute_hash()

        target_stage = "Production" if promote_to_production else "Staging"
        alias = "champion" if promote_to_production else "challenger"

        tags = {
            "git_commit": git_commit,
            "dataset_version_hash": data_hash,
            "config_hash": config_hash,
            "primary_metric": gate_decision.get("primary_metric", "f1_macro"),
            "candidate_metric_value": str(gate_decision.get("candidate_metric_value", "")),
            "gate_passed": "true",
            "stage": target_stage.lower(),
            "alias": alias,
        }

        # Log metrics as individual tags
        for m_key, m_val in gate_decision.get("metrics", {}).items():
            tags[m_key] = str(round(m_val, 4))

        for k, v in tags.items():
            client.set_model_version_tag(name=model_name, version=version_id, key=k, value=v)

        # Set modern MLflow alias (@champion / @challenger)
        try:
            client.set_registered_model_alias(name=model_name, alias=alias, version=version_id)
            logger.info(
                "Assigned model alias",
                alias=alias,
                version=version_id,
                model_name=model_name,
            )
        except Exception as alias_err:
            logger.debug("Model alias assignment skipped or not supported", error=str(alias_err))

        # Assign Stage: Staging by default, or Production if requested
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                client.transition_model_version_stage(
                    name=model_name,
                    version=version_id,
                    stage=target_stage,
                    archive_existing_versions=(target_stage == "Production"),
                )
            logger.info(
                f"Transitioned model version to '{target_stage}'",
                version=version_id,
                model_name=model_name,
            )
        except Exception as stage_err:
            logger.warning(
                "Standard stage transition unavailable; tag will reflect stage status",
                error=str(stage_err),
            )
            client.set_model_version_tag(
                name=model_name, version=version_id, key="stage", value=target_stage.lower()
            )

        registration_result = {
            "model_name": model_name,
            "model_version": version_id,
            "run_id": run_id,
            "target_stage": target_stage,
            "tags": tags,
        }

        # Save registration receipt
        receipt_path = config.train.checkpoint_dir / "registration_receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(registration_result, f, indent=2)

        return registration_result

    except Exception as e:
        if isinstance(e, ModelRegistrationError):
            raise
        raise ModelRegistrationError(
            f"Failed during MLflow model registration: {e}",
            {"model_name": model_name, "run_id": run_id, "error": str(e)},
        ) from e


@click.command()
@click.option(
    "--config", "config_path", default="configs/config.yaml", help="Path to master config YAML"
)
@click.option(
    "--promote-to-production",
    is_flag=True,
    default=False,
    help="Explicitly promote this model version to Production stage",
)
def main(config_path: str, promote_to_production: bool) -> None:
    """CLI entrypoint for Stage 5: Model Registration."""
    try:
        run_model_registration(config_path=config_path, promote_to_production=promote_to_production)
    except ModelRegistrationError as e:
        logger.error("Model registration failed!", error=str(e))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
