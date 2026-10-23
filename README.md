# NEU-DET Steel Surface Defect Classification

A PyTorch CNN that classifies grayscale steel-surface images into six defect categories: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`.

## Problem

Automatically identify common surface defects in steel inspection images. This repository contains the model, training/evaluation/prediction scripts, and a checkpoint suitable for a live demonstration. It is an educational baseline, not a production inspection system.

## Dataset

This project uses the NEU Surface Defect Database (NEU-DET), containing six classes of grayscale steel-strip surface defects. Dataset images are not copied into this Git repository. Obtain the dataset from its official/source distribution and arrange it as:

```text
NEU-DET/
├── train/
│   └── images/
│       ├── crazing/
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/
    └── images/
        └── [same six class folders]
```

The training script uses an 80/20 random train/validation split of `train/images`. `validation/images` is held aside for final evaluation. Images are converted to grayscale, resized to 200×200, converted to tensors, and normalized with mean and standard deviation 0.5.

## Architecture

The CNN has three convolution blocks:

- Conv2d: 1→16, ReLU, MaxPool2d
- Conv2d: 16→32, ReLU, MaxPool2d
- Conv2d: 32→64, ReLU, MaxPool2d
- Flatten → Linear(64×25×25, 128) → ReLU → Dropout(0.3) → Linear(128, 6)

The classifier returns six logits and is trained with cross-entropy loss and Adam.

## Results

The original notebook run recorded **91.11% (328/360)** on the provided validation/test images, but its weights were not saved and that exact model cannot be restored from the accuracy number alone. The included checkpoint is a newly trained model saved at its best training-holdout epoch (epoch 17, **96.53%** on the 20% holdout). It scores **86.67% (312/360)** on the supplied 360-image validation split. Since this validation split has been used repeatedly to compare experiments, treat that score as exploratory rather than as an unbiased final benchmark. Results vary with the split, random seed, software versions, and hardware.

## Setup

Use Python 3.10–3.13. From this repository directory:

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch and torchvision versions matching your CUDA setup using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

## Train

From the repository root, point `--data-dir` to the directory containing `train/` and `validation/`:

```bash
python src/train.py --data-dir "../NEU-DET" --epochs 20 --batch-size 32 --learning-rate 0.001
```

The best-validation-accuracy checkpoint is written to `models/neu_det_cnn.pth` by default.

## Evaluate

```bash
python src/evaluate.py --data-dir "../NEU-DET" --checkpoint models/neu_det_cnn.pth
```

## Predict one image

```bash
python src/predict.py "../NEU-DET/validation/images/scratches/example.jpg" --checkpoint models/neu_det_cnn.pth
```

## Repository contents

```text
neu-det-cnn/
├── models/neu_det_cnn.pth
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```
