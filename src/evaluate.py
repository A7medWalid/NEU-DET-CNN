"""Evaluate a saved checkpoint on the official NEU-DET validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import build_transform, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="NEU-DET dataset root")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/neu_det_cnn.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    test_dir = args.data_dir / "validation" / "images"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test images folder not found: {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(str(args.checkpoint), device)
    test_dataset = datasets.ImageFolder(test_dir, transform=build_transform())
    if test_dataset.classes != class_names:
        raise ValueError(
            "Dataset class order does not match checkpoint. "
            f"Checkpoint={class_names}; data={test_dataset.classes}"
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
