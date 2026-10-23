"""CNN architecture and preprocessing for NEU-DET surface defect classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms

IMAGE_SIZE = 200
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)


def build_transform() -> transforms.Compose:
    """Create the preprocessing pipeline expected by the trained model."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


class CNN(nn.Module):
    """Three-block CNN for six NEU-DET defect classes."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
) -> tuple[CNN, list[str]]:
    """Load a saved state-dict checkpoint and return model plus class names."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = CNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names
