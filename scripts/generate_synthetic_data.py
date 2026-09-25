"""Utility script to generate synthetic test dataset for local development and CI testing."""

from pathlib import Path

import click
import numpy as np
from PIL import Image


def generate_synthetic_image(
    output_path: Path, class_idx: int, size: tuple[int, int] = (224, 224)
) -> None:
    """Generate a synthetic retina-like circular image with class-specific characteristics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Base noise
    arr = np.random.randint(20, 60, (size[0], size[1], 3), dtype=np.uint8)

    # Circular mask simulating fundus camera field of view
    y, x = np.ogrid[: size[0], : size[1]]
    center_y, center_x = size[0] / 2, size[1] / 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(size) / 2 - 10) ** 2

    # Tint circular area orange/reddish (retinal background)
    arr[mask, 0] = np.random.randint(140, 210, size=np.sum(mask))  # Red
    arr[mask, 1] = np.random.randint(60, 110, size=np.sum(mask))  # Green
    arr[mask, 2] = np.random.randint(20, 50, size=np.sum(mask))  # Blue

    # Add distinguishing pattern for class 1 (e.g. referable DR exudates / spots)
    if class_idx == 1:
        num_spots = np.random.randint(3, 8)
        for _ in range(num_spots):
            spot_y = np.random.randint(int(size[0] * 0.3), int(size[0] * 0.7))
            spot_x = np.random.randint(int(size[1] * 0.3), int(size[1] * 0.7))
            radius = np.random.randint(3, 7)
            spot_mask = (x - spot_x) ** 2 + (y - spot_y) ** 2 <= radius**2
            arr[spot_mask & mask] = [240, 240, 160]  # Bright yellow exudate lesion

    img = Image.fromarray(arr)
    img.save(output_path, "JPEG", quality=90)


@click.command()
@click.option("--output-dir", default="data/raw", help="Directory where raw data should be created")
@click.option("--samples-per-class", default=25, help="Number of images per class")
def main(output_dir: str, samples_per_class: int) -> None:
    """Generate synthetic image classification dataset."""
    base_dir = Path(output_dir)
    classes = ["healthy", "referable_dr"]

    print(f"Generating synthetic dataset in {base_dir} ({samples_per_class} per class)...")
    for idx, cls_name in enumerate(classes):
        cls_dir = base_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img_path = cls_dir / f"sample_{i:04d}.jpg"
            generate_synthetic_image(img_path, class_idx=idx)

    print("Synthetic dataset generated successfully!")


if __name__ == "__main__":
    main()
