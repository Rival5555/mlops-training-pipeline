"""Stage 3: Config-Driven PyTorch Model Training with MLflow Tracking.

Features:
- Config-driven training parameters (learning rate, epochs, optimizer, architecture, transforms)
- Mixed precision training (AMP autocast) and gradient accumulation
- Early stopping with configurable patience and metric/mode
- Checkpointing both 'best' (by val metric) and 'last' models
- Live logging to MLflow: parameters, training & validation metrics per epoch, confusion matrix, and checkpoint artifacts
"""

import json
import time
from pathlib import Path
from typing import Any

import click
import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.pipeline.common.exceptions import TrainingError
from src.pipeline.common.logger import get_logger
from src.pipeline.common.mlflow_utils import get_git_commit_hash, setup_mlflow
from src.pipeline.config.schema import PipelineConfig, load_config

logger = get_logger("s03_train")


class ManifestImageDataset(Dataset):
    """PyTorch Dataset loading images declared in split_manifest.json."""

    def __init__(
        self, records: list[dict[str, Any]], root_dir: Path, transform: transforms.Compose
    ) -> None:
        self.records = records
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        file_path = self.root_dir / Path(record["filepath"])
        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                tensor_data = self.transform(img_rgb)
                return tensor_data, int(record["label"])
        except Exception as e:
            raise TrainingError(f"Failed to load image at {file_path}: {e}") from e


def build_transforms(config: PipelineConfig, is_train: bool) -> transforms.Compose:
    """Build transform pipeline declaratively from config."""
    norm = config.data.normalization
    aug = config.data.augmentation
    img_size = config.data.image_size

    if is_train:
        transform_list = [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=aug.horizontal_flip_prob),
            transforms.RandomRotation(degrees=aug.rotation_degrees),
            transforms.ColorJitter(
                brightness=aug.color_jitter_brightness,
                contrast=aug.color_jitter_contrast,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.mean, std=norm.std),
        ]
    else:
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.mean, std=norm.std),
        ]
    return transforms.Compose(transform_list)


def create_model(config: PipelineConfig) -> nn.Module:
    """Instantiate model architecture and customize classification head."""
    arch = config.model.architecture.lower()
    num_classes = config.model.num_classes
    pretrained = config.model.pretrained
    dropout = config.model.dropout_rate

    if "efficientnet" in arch:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, num_classes),
        )
    elif "resnet18" in arch:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif "resnet50" in arch:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise TrainingError(f"Unsupported architecture: {arch}")

    if config.model.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze head
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True

    return model


def run_training(
    config: PipelineConfig | None = None, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Execute model training loop with MLflow tracking and checkpointing."""
    start_time = time.time()
    if config is None:
        config = load_config(config_path)

    # Set seeds for deterministic training
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    processed_dir = config.data.processed_data_dir
    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise TrainingError(f"Split manifest missing at {manifest_path}. Run Stage 2 first.")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Prepare datasets and loaders
    root_dir = config.data.raw_data_dir.parent
    train_ds = ManifestImageDataset(
        manifest["records"]["train"], root_dir, build_transforms(config, is_train=True)
    )
    val_ds = ManifestImageDataset(
        manifest["records"]["val"], root_dir, build_transforms(config, is_train=False)
    )

    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Device & Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Initializing training",
        device=str(device),
        train_samples=len(train_ds),
        val_samples=len(val_ds),
    )

    model = create_model(config).to(device)

    # Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    optimizer_name = config.train.optimizer.lower()
    lr = config.train.learning_rate
    weight_decay = config.train.weight_decay

    optimizer: torch.optim.Optimizer
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise TrainingError(f"Unsupported optimizer: {optimizer_name}")

    epochs = config.train.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision scaler
    use_amp = config.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = config.train.gradient_accumulation_steps

    # Checkpoint setup
    ckpt_dir = config.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pt"
    last_model_path = ckpt_dir / "last_model.pt"

    # MLflow Setup
    setup_mlflow(config.mlflow)
    git_commit = get_git_commit_hash()
    config_hash = config.compute_hash()
    data_hash = manifest.get("dataset_version_hash", "unknown")

    best_val_metric = float("inf") if config.train.early_stopping_mode == "min" else float("-inf")
    patience_counter = 0
    best_epoch = 0

    with mlflow.start_run(run_name=f"{config.model.architecture}-run") as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run", run_id=run_id)

        # Log parameters
        mlflow.log_params(
            {
                "architecture": config.model.architecture,
                "num_classes": config.model.num_classes,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "optimizer": optimizer_name,
                "scheduler": config.train.scheduler,
                "mixed_precision": use_amp,
                "gradient_accumulation_steps": grad_accum_steps,
                "git_commit": git_commit,
                "config_hash": config_hash,
                "dataset_version_hash": data_hash,
            }
        )
        mlflow.set_tags(
            {
                "stage": "training",
                "git_commit": git_commit,
                "config_hash": config_hash,
                "data_hash": data_hash,
            }
        )

        for epoch in range(1, epochs + 1):
            # --- Training Loop ---
            model.train()
            train_loss = 0.0
            train_preds: list[int] = []
            train_targets: list[int] = []

            optimizer.zero_grad()
            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss_to_backward = loss / grad_accum_steps

                scaler.scale(loss_to_backward).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                train_preds.extend(preds)
                train_targets.extend(targets.cpu().numpy())

            scheduler.step()

            epoch_train_loss = train_loss / len(train_ds)
            if np.isnan(epoch_train_loss) or np.isinf(epoch_train_loss):
                raise TrainingError(
                    f"Training loss diverged to {epoch_train_loss} at epoch {epoch}"
                )
            if np.isnan(epoch_train_loss) or np.isinf(epoch_train_loss):
                raise TrainingError(
                    f"Training loss diverged to {epoch_train_loss} at epoch {epoch}"
                )
            epoch_train_acc = accuracy_score(train_targets, train_preds)
            epoch_train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

            # --- Validation Loop ---
            model.eval()
            val_loss = 0.0
            val_preds: list[int] = []
            val_targets: list[int] = []

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                    val_loss += loss.item() * images.size(0)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(targets.cpu().numpy())

            epoch_val_loss = val_loss / len(val_ds)
            epoch_val_acc = accuracy_score(val_targets, val_preds)
            epoch_val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": epoch_train_loss,
                    "train_accuracy": epoch_train_acc,
                    "train_f1_macro": epoch_train_f1,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": epoch_val_acc,
                    "val_f1_macro": epoch_val_f1,
                    "lr": float(scheduler.get_last_lr()[0]),
                },
                step=epoch,
            )

            logger.info(
                f"Epoch {epoch}/{epochs}",
                train_loss=round(epoch_train_loss, 4),
                val_loss=round(epoch_val_loss, 4),
                val_acc=round(epoch_val_acc, 4),
                val_f1=round(epoch_val_f1, 4),
            )

            # Checkpoint: Last Model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": epoch_val_loss,
                    "val_f1": epoch_val_f1,
                    "config": config.model_dump(),
                    "run_id": run_id,
                },
                last_model_path,
            )

            # Checkpoint: Best Model & Early Stopping
            monitored_val = (
                epoch_val_loss if config.train.early_stopping_metric == "val_loss" else epoch_val_f1
            )
            is_better = (
                (monitored_val < best_val_metric)
                if config.train.early_stopping_mode == "min"
                else (monitored_val > best_val_metric)
            )

            if is_better:
                best_val_metric = monitored_val
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": epoch_val_loss,
                        "val_accuracy": epoch_val_acc,
                        "val_f1": epoch_val_f1,
                        "config": config.model_dump(),
                        "run_id": run_id,
                        "dataset_version_hash": data_hash,
                        "config_hash": config_hash,
                        "git_commit": git_commit,
                    },
                    best_model_path,
                )
                logger.info(
                    "Saved new best checkpoint", epoch=epoch, metric=round(monitored_val, 4)
                )
            else:
                patience_counter += 1
                if patience_counter >= config.train.early_stopping_patience:
                    logger.info("Early stopping triggered", epoch=epoch, patience=patience_counter)
                    break

        # Save training summary and MLflow artifacts
        duration = round(time.time() - start_time, 2)
        mlflow.log_metric("training_duration_seconds", duration)
        mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")

        # Save training info metadata
        train_info = {
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_metric_value": best_val_metric,
            "training_duration_sec": duration,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "config_hash": config_hash,
            "dataset_version_hash": data_hash,
            "git_commit": git_commit,
        }
        info_path = ckpt_dir / "training_summary.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(train_info, f, indent=2)

        mlflow.log_artifact(str(info_path), artifact_path="metadata")

        logger.info("Training stage completed", run_id=run_id, duration_sec=duration)
        return train_info


@click.command()
@click.option(
    "--config", "config_path", default="configs/config.yaml", help="Path to master config YAML"
)
def main(config_path: str) -> None:
    """CLI entrypoint for Stage 3: Training."""
    try:
        run_training(config_path=config_path)
    except TrainingError as e:
        logger.error("Training stage failed!", error=str(e))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
