"""Predict the defect class of one image with a saved NEU-DET model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model import build_transform, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/neu_det_cnn.pth"))
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(str(args.checkpoint), device)
    image = Image.open(args.image).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_id = int(probabilities.argmax().item())

    print(f"Prediction: {class_names[class_id]}")
    print(f"Confidence: {probabilities[class_id].item():.2%}")


if __name__ == "__main__":
    main()
