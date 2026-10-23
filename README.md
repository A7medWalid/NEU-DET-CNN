# NEU-DET-CNN

### Steel Surface Defect Classification Using a Custom CNN

A PyTorch-based convolutional neural network (CNN) that classifies grayscale steel-surface images into six defect categories. This project is an educational baseline for exploring deep learning in industrial quality inspection; it is not a production-certified inspection system.

## Problem

Visible defects in steel surfaces can affect product quality and manufacturing efficiency. This project explores how a CNN can automatically classify steel-strip images and support more consistent inspection.

## Dataset

The project uses the **NEU Surface Defect Database (NEU-DET)**, which contains six defect classes:

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

Dataset images are not included in this repository. Download the dataset from its source and arrange it as follows:

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
        ├── crazing/
        ├── inclusion/
        ├── patches/
        ├── pitted_surface/
        ├── rolled-in_scale/
        └── scratches/
```

The training script creates an 80/20 random training/validation split from `train/images`. The separate `validation/images` directory is used for evaluation. Images are converted to grayscale, resized to **200 × 200** pixels, converted to tensors, and normalized using a mean and standard deviation of `0.5`.

## Model architecture

The custom CNN contains three convolutional blocks followed by a classifier:

1. `Conv2d(1, 16)` → ReLU → Max Pooling
2. `Conv2d(16, 32)` → ReLU → Max Pooling
3. `Conv2d(32, 64)` → ReLU → Max Pooling
4. Flatten
5. `Linear(64 × 25 × 25, 128)` → ReLU → Dropout (`0.3`)
6. `Linear(128, 6)` output layer

The model produces six class scores and is trained using cross-entropy loss and the Adam optimizer.

## Results

The available runs produced the following results:

| Evaluation | Accuracy | Notes |
|---|---:|---|
| Earlier notebook run | 91.11% (328/360) | The weights from this run were not saved, so this exact model cannot be restored from the score alone. |
| Training holdout | 96.53% | The included checkpoint was saved at its best holdout epoch (epoch 17). |
| Supplied 360-image validation split | 86.67% (312/360) | Exploratory result; this split has been used repeatedly to compare experiments. |

**Interpretation:** Treat these scores as experimental results, not as a final unbiased benchmark. The 360-image split has been used during model development, so an independent test set is needed for a fair final evaluation. Accuracy may vary with the random split, seed, software versions, and hardware.

## Requirements

- Python 3.10–3.13
- PyTorch
- torchvision
- NumPy
- Pillow
- scikit-learn
- Matplotlib

Install the listed Python packages from the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install the PyTorch and torchvision builds compatible with your CUDA setup by following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Train

From the repository root, set `--data-dir` to the directory that contains `train/` and `validation/`:

```bash
python src/train.py --data-dir "../NEU-DET" --epochs 20 --batch-size 32 --learning-rate 0.001
```

By default, the best-validation-accuracy checkpoint is saved to:

```text
models/neu_det_cnn.pth
```

## Evaluate

```bash
python src/evaluate.py --data-dir "../NEU-DET" --checkpoint models/neu_det_cnn.pth
```

## Predict an image

Replace the example path with the path to an image you want to classify:

```bash
python src/predict.py "../NEU-DET/validation/images/scratches/example.jpg" --checkpoint models/neu_det_cnn.pth
```

## Repository structure

```text
NEU-DET-CNN/
├── models/
│   └── neu_det_cnn.pth
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Limitations and future work

- Evaluate on a genuinely untouched test set.
- Report class-wise precision, recall, F1-score, and a confusion matrix.
- Test predictions on additional steel-surface images collected under different conditions.
- Investigate data augmentation and methods to reduce overfitting.
- Measure inference speed before considering real-time or conveyor-belt integration.

## Technologies

Python · PyTorch · torchvision · NumPy · Pillow · scikit-learn · Matplotlib
