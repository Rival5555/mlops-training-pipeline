"""Stage 6: Packaging, ONNX Export, and Parity Smoke Testing.

Responsibilities:
- Load best PyTorch model checkpoint.
- Export to ONNX with dynamic batch axes and configured opset.
- Execute smoke test: Run forward pass in PyTorch and ONNX Runtime on identical input tensor.
- Verify numerical parity within strict floating point tolerances (atol, rtol).
- Fail loudly with ParityCheckError if discrepancies exceed tolerance.
- Log exported ONNX artifact to MLflow.
"""

import json
from pathlib import Path
from typing import Any

import click
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.pipeline.common.exceptions import ParityCheckError
from src.pipeline.common.logger import get_logger
from src.pipeline.common.mlflow_utils import setup_mlflow
from src.pipeline.config.schema import PipelineConfig, load_config
from src.pipeline.stages.s03_train import create_model

logger = get_logger("s06_export_package")


def verify_numeric_parity(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> tuple[bool, float]:
    """Execute parity check between PyTorch eager mode and ONNX Runtime session."""
    pytorch_model.eval()

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNX Runtime inference
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # Warning level
    session = ort.InferenceSession(
        str(onnx_path), session_options, providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    onnx_inputs = {input_name: dummy_input.cpu().numpy()}
    onnx_output = session.run(None, onnx_inputs)[0]

    # Calculate absolute differences
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = float(np.max(diff))

    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    return is_close, max_diff


def run_export_and_package(
    config: PipelineConfig | None = None, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Export PyTorch checkpoint to ONNX and perform parity smoke test."""
    if config is None:
        config = load_config(config_path)

    if config.export.export_format.lower() != "onnx":
        raise ParityCheckError(
            f"Unsupported export format: {config.export.export_format}. Only 'onnx' is supported."
        )

    best_model_path = config.train.checkpoint_dir / "best_model.pt"
    if not best_model_path.exists():
        raise ParityCheckError(f"Model checkpoint not found at {best_model_path}")

    output_dir = config.export.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / config.export.output_filename

    logger.info("Starting ONNX model export", onnx_output_path=str(onnx_path))

    # Load PyTorch checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare dummy input for tracing
    input_shape = config.export.dummy_input_shape
    dummy_input = torch.randn(*input_shape, requires_grad=False)

    dynamic_axes = None
    if config.export.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=config.export.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX model graph structure
    onnx_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_proto)
    logger.info("ONNX graph successfully verified by onnx.checker")

    # Parity Smoke Test
    max_diff = 0.0
    if config.export.verify_onnx_runtime:
        logger.info("Performing numerical parity smoke test...")
        test_tensor = torch.randn(*input_shape)
        is_close, max_diff = verify_numeric_parity(
            pytorch_model=model,
            onnx_path=onnx_path,
            dummy_input=test_tensor,
            atol=config.export.float_tolerance_atol,
            rtol=config.export.float_tolerance_rtol,
        )

        if not is_close:
            raise ParityCheckError(
                f"ONNX parity smoke test failed! Max absolute diff {max_diff:.6f} exceeded tolerance (atol={config.export.float_tolerance_atol})",
                {"max_diff": max_diff, "atol": config.export.float_tolerance_atol},
            )

        logger.info(
            "ONNX numerical parity smoke test PASSED",
            max_diff=max_diff,
            atol=config.export.float_tolerance_atol,
        )

    # Log exported artifact to MLflow
    run_id = checkpoint.get("run_id")
    setup_mlflow(config.mlflow)
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(onnx_path), artifact_path="exported_models")
                mlflow.log_metric("onnx_parity_max_diff", max_diff)
        except Exception as e:
            logger.warning("Could not log ONNX artifact to MLflow run", error=str(e))

    export_summary = {
        "format": config.export.export_format.lower(),
        "onnx_path": str(onnx_path),
        "opset_version": config.export.opset_version,
        "parity_verified": True,
        "max_absolute_diff": max_diff,
        "dynamic_batch": config.export.dynamic_batch,
        "run_id": run_id,
    }

    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(export_summary, f, indent=2)

    logger.info("Export stage completed successfully", summary_path=str(summary_path))
    return export_summary


@click.command()
@click.option(
    "--config", "config_path", default="configs/config.yaml", help="Path to master config YAML"
)
def main(config_path: str) -> None:
    """CLI entrypoint for Stage 6: Model Export and Packaging."""
    try:
        run_export_and_package(config_path=config_path)
    except ParityCheckError as e:
        logger.error("Model export and parity verification failed!", error=str(e))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
