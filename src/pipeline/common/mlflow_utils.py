"""MLflow utilities for experiment tracking, artifact logging, and model registry management."""

import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.pipeline.common.logger import get_logger
from src.pipeline.config.schema import MLflowConfig

logger = get_logger("mlflow_utils")


def get_git_commit_hash() -> str:
    """Safely get the current git commit hash, or return 'unknown' if not in git."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return os.getenv("GITHUB_SHA", "unknown_commit")


def setup_mlflow(config: MLflowConfig) -> MlflowClient:
    """Initialize MLflow tracking URI and experiment, returning an active MlflowClient."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = config.tracking_uri
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
        logger.info("Connected to MLflow tracking backend", tracking_uri=tracking_uri)
    except Exception as e:
        local_uri = "sqlite:///mlflow.db"
        logger.warning(
            "Could not connect to tracking server; falling back to local SQLite backend",
            error=str(e),
            fallback_uri=local_uri,
        )
        mlflow.set_tracking_uri(local_uri)
        client = MlflowClient(tracking_uri=local_uri)

    mlflow.set_experiment(config.experiment_name)
    return client


def get_production_model_metrics(
    client: MlflowClient, model_name: str, metric_name: str
) -> float | None:
    """Retrieve the primary metric of the currently active 'production' tagged model version.

    Returns None if no production model exists.
    """
    try:
        # Search registered model versions
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [
            v
            for v in versions
            if (v.current_stage and v.current_stage.lower() == "production")
            or (v.tags and v.tags.get("stage", "").lower() == "production")
        ]

        if not prod_versions:
            logger.info(
                "No existing model tagged as production found in registry", model_name=model_name
            )
            return None

        # Sort by creation timestamp descending to get the newest production model
        prod_version: ModelVersion = sorted(
            prod_versions, key=lambda x: x.creation_timestamp, reverse=True
        )[0]

        if not prod_version.run_id:
            return None

        # Retrieve the run that created this model version
        run = client.get_run(prod_version.run_id)
        metric_val = run.data.metrics.get(metric_name)

        if metric_val is None:
            # Check tags for metric
            if prod_version.tags and metric_name in prod_version.tags:
                metric_val = float(prod_version.tags[metric_name])

        logger.info(
            "Retrieved production baseline",
            model_name=model_name,
            version=prod_version.version,
            metric_name=metric_name,
            value=metric_val,
        )
        return metric_val
    except Exception as e:
        logger.warning("Error fetching production baseline from registry", error=str(e))
        return None


def log_confusion_matrix(
    confusion_matrix_data: list[list[int]], class_names: list[str], output_path: Path
) -> Path:
    """Generate and save a visual confusion matrix figure for MLflow artifact logging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion_matrix_data, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Text annotations
    for i in range(len(confusion_matrix_data)):
        for j in range(len(confusion_matrix_data[i])):
            ax.text(
                j,
                i,
                str(confusion_matrix_data[i][j]),
                ha="center",
                va="center",
                color="white"
                if confusion_matrix_data[i][j] > (max(map(max, confusion_matrix_data)) / 2)
                else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
