"""Stage 4: Held-out Test Set Evaluation and Automated Production Validation Gate.

Responsibilities:
- Evaluate candidate best checkpoint on held-out test split.
- Compute comprehensive metrics: accuracy, macro/weighted F1, per-class precision/recall, confusion matrix.
- Query MLflow Model Registry for the active 'production' tagged model version.
- Validate whether candidate beats production by configured delta (e.g. F1 + 0.01) and meets absolute minimum.
- Log artifacts and gate metrics back to MLflow.
- Fail loudly with EvaluationGateError if gate criteria are not satisfied.
"""

import json
from pathlib import Path
from typing import Any

import click
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.pipeline.common.exceptions import EvaluationGateError
from src.pipeline.common.logger import get_logger
from src.pipeline.common.mlflow_utils import (
    get_production_model_metrics,
    log_confusion_matrix,
    setup_mlflow,
)
from src.pipeline.config.schema import PipelineConfig, load_config
from src.pipeline.stages.s03_train import ManifestImageDataset, build_transforms, create_model

logger = get_logger("s04_eval_gate")


def evaluate_model_on_dataset(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], list[list[float]]]:
    """Execute inference loop to extract predictions, targets, and probabilities."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return all_targets, all_preds, all_probs


def check_promotion_gate(
    candidate_metric: float,
    production_metric: float | None,
    min_absolute_threshold: float,
    min_improvement: float,
    allow_first_model: bool = True,
) -> tuple[bool, str, float]:
    """Pure evaluation logic to determine whether a model passes the production gate.

    Returns:
        (passed, reason, required_threshold)
    """
    if candidate_metric < min_absolute_threshold:
        return (
            False,
            f"Candidate metric ({candidate_metric:.4f}) failed to reach absolute minimum floor ({min_absolute_threshold:.4f})",
            min_absolute_threshold,
        )

    # First model in registry edge-case
    if production_metric is None:
        if allow_first_model:
            return (
                True,
                f"No existing production model baseline found. Candidate passed absolute threshold ({min_absolute_threshold:.4f})",
                min_absolute_threshold,
            )
        return (
            False,
            "No existing production model baseline found and allow_first_model is False",
            min_absolute_threshold,
        )

    # Calculate required threshold, capped at 1.0 (since normalized metrics cannot exceed 1.0)
    required_threshold = min(1.0, production_metric + min_improvement)

    # If production baseline is already at maximum ceiling, allow candidate if it also achieves maximum ceiling
    if production_metric >= (1.0 - min_improvement) and candidate_metric >= (1.0 - 1e-4):
        return (
            True,
            f"Candidate metric ({candidate_metric:.4f}) reached maximum quality ceiling alongside production baseline ({production_metric:.4f})",
            required_threshold,
        )

    if candidate_metric >= required_threshold:
        delta = candidate_metric - production_metric
        return (
            True,
            f"Candidate metric ({candidate_metric:.4f}) beat production ({production_metric:.4f}) "
            f"by +{delta:.4f} (exceeds required +{min_improvement:.4f})",
            required_threshold,
        )

    delta = candidate_metric - production_metric
    return (
        False,
        f"Candidate metric ({candidate_metric:.4f}) failed to beat production ({production_metric:.4f}) "
        f"by required +{min_improvement:.4f} (actual delta: {delta:+.4f})",
        required_threshold,
    )


def run_evaluation_gate(
    config: PipelineConfig | None = None, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Execute evaluation on test set and enforce promotion gate."""
    if config is None:
        config = load_config(config_path)

    # Check dependencies
    ckpt_dir = config.train.checkpoint_dir
    best_model_path = ckpt_dir / "best_model.pt"
    if not best_model_path.exists():
        raise EvaluationGateError(f"Best model checkpoint missing at {best_model_path}")

    manifest_path = config.data.processed_data_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise EvaluationGateError(f"Split manifest missing at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Build Test DataLoader
    root_dir = config.data.raw_data_dir.parent
    test_ds = ManifestImageDataset(
        manifest["records"]["test"], root_dir, build_transforms(config, is_train=False)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Evaluating candidate model on test set", test_samples=len(test_ds), device=str(device)
    )

    # Load Model Checkpoint
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run inference
    y_true, y_pred, y_prob = evaluate_model_on_dataset(model, test_loader, device)

    # Compute Metrics
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    all_class_indices = list(range(len(config.model.class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_class_indices).tolist()

    computed_metrics = {
        "test_accuracy": acc,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "test_precision_macro": prec_macro,
        "test_recall_macro": rec_macro,
    }

    class_rep = classification_report(
        y_true, y_pred, target_names=config.model.class_names, output_dict=True, zero_division=0
    )

    # Connect to MLflow
    client = setup_mlflow(config.mlflow)
    primary_metric_key = config.evaluate.primary_metric
    candidate_metric = computed_metrics.get(
        f"test_{primary_metric_key}", computed_metrics.get(primary_metric_key, f1_macro)
    )

    # Fetch active production baseline
    prod_baseline = get_production_model_metrics(
        client=client,
        model_name=config.mlflow.model_name,
        metric_name=f"test_{primary_metric_key}",
    )

    # Automated Gate Check
    passed, reason, required_threshold = check_promotion_gate(
        candidate_metric=candidate_metric,
        production_metric=prod_baseline,
        min_absolute_threshold=config.evaluate.min_absolute_threshold,
        min_improvement=config.evaluate.min_improvement_over_production,
        allow_first_model=config.evaluate.allow_first_model,
    )

    logger.info(
        "Gate decision computed",
        passed=passed,
        candidate_metric=candidate_metric,
        production_baseline=prod_baseline,
        required_threshold=required_threshold,
        reason=reason,
    )

    # Output decision payload
    gate_decision = {
        "passed": passed,
        "primary_metric": primary_metric_key,
        "candidate_metric_value": candidate_metric,
        "production_baseline_value": prod_baseline,
        "required_threshold": required_threshold,
        "min_improvement_required": config.evaluate.min_improvement_over_production,
        "reason": reason,
        "metrics": computed_metrics,
        "classification_report": class_rep,
        "run_id": checkpoint.get("run_id"),
    }

    # Save decision file
    output_path = config.evaluate.gate_decision_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gate_decision, f, indent=2)

    # Save and Log Confusion Matrix
    cm_plot_path = ckpt_dir / "confusion_matrix.png"
    log_confusion_matrix(cm, config.model.class_names, cm_plot_path)

    # Log to MLflow if run_id exists
    run_id = checkpoint.get("run_id")
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                for k, v in computed_metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.log_metric("gate_passed", 1 if passed else 0)
                mlflow.log_artifact(str(output_path), artifact_path="evaluation")
                mlflow.log_artifact(str(cm_plot_path), artifact_path="evaluation")
        except Exception as e:
            logger.warning("Could not log gate artifacts to existing MLflow run", error=str(e))

    # Enforce Gate Strategy
    if not passed:
        if config.evaluate.gate_strategy.value == "fail":
            raise EvaluationGateError(
                f"Automated Validation Gate FAILED: {reason}",
                {"gate_decision": gate_decision},
            )
        logger.warning(
            f"Gate failed but gate_strategy is '{config.evaluate.gate_strategy.value}': {reason}"
        )

    return gate_decision


@click.command()
@click.option(
    "--config", "config_path", default="configs/config.yaml", help="Path to master config YAML"
)
def main(config_path: str) -> None:
    """CLI entrypoint for Stage 4: Evaluation and Validation Gate."""
    try:
        run_evaluation_gate(config_path=config_path)
    except EvaluationGateError as e:
        logger.error("Evaluation Gate check rejected candidate model!", error=str(e))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
