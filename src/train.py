"""Train the NEU-DET CNN and save the checkpoint with the best validation accuracy."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from model import CNN, build_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="NEU-DET directory containing train/images and validation/images",
    )
    parser.add_argument("--output", type=Path, default=Path("models/neu_det_cnn.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_dir = args.data_dir / "train" / "images"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training images folder not found: {train_dir}")

    full_train = datasets.ImageFolder(train_dir, transform=build_transform())
    if len(full_train.classes) != 6:
        raise ValueError(f"Expected 6 classes, found {len(full_train.classes)}: {full_train.classes}")

    validation_size = max(1, int(round(0.2 * len(full_train))))
    training_size = len(full_train) - validation_size
    # Match the notebook's seeded split order: random_split consumes the global
    # seeded RNG before the model is initialized, reproducing its initial weights.
    train_subset, val_subset = random_split(full_train, [training_size, validation_size])

    # Validation uses the same source images with the same deterministic transform.
    val_dataset = Subset(full_train, val_subset.indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(full_train.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss / train_total:.4f}, "
            f"train acc {train_accuracy:.4f} | "
            f"val loss {val_loss / val_total:.4f}, "
            f"val acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "class_names": full_train.classes,
                    "image_size": 200,
                    "normalize_mean": [0.5],
                    "normalize_std": [0.5],
                    "best_val_accuracy": best_val_accuracy,
                    "epoch": epoch,
                    "architecture": "NEU-DET-CNN-v1",
                },
                args.output,
            )
            print(f"  Saved best checkpoint: {args.output}")

    print(f"Best validation accuracy: {best_val_accuracy:.2%}")


if __name__ == "__main__":
    main()
