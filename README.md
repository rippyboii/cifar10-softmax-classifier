# cifar10-softmax-classifier

A single-layer softmax classifier trained on CIFAR-10 using mini-batch gradient descent with L2 regularization. Built from scratch with NumPy as part of **DD2424 Deep Learning in Data Science** (Assignment 1) at KTH.

## Overview

The model learns a linear mapping from 3072-dimensional image vectors (32×32×3) to a 10-class probability distribution:

```
s = Wx + b
p = softmax(s)
```

Training minimizes the cross-entropy loss plus an L₂ penalty on the weight matrix:

```
J = (1/n) Σ -log(p_y) + λ‖W‖²
```

## Project Structure

```
cifar10-softmax-classifier/
├── src/
│   └── main.py              # All model code: loading, training, evaluation, plotting
├── figures/                  # Generated training curves and weight visualizations
├── Datasets/
│   └── cifar-10-python/
│       └── cifar-10-batches-py/
│           ├── data_batch_1  # Training data (10,000 images)
│           ├── data_batch_2  # Validation data (10,000 images)
│           └── test_batch    # Test data (10,000 images)
└── README.md
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- PyTorch (CPU only, used for gradient verification)

Install dependencies:

```bash
pip install numpy matplotlib torch
```

## Dataset Setup
1. Either use the given dataset in the codebase
2. OR
Download the CIFAR-10 Python version from [the official site](https://www.cs.toronto.edu/~kriz/cifar.html), then extract it into the `Datasets` directory:

## Usage

```bash
cd src
python main.py
```

This will:

1. Load and normalize CIFAR-10 data (per-dimension zero mean, unit variance)
2. Verify analytic gradients against PyTorch on a small subset
3. Train the classifier under four hyperparameter settings
4. Save loss/cost curves and weight template visualizations to `figures/`

## Implemented Functions

| Function | Description |
|---|---|
| `LoadBatch` | Reads a CIFAR-10 pickle file, returns image matrix (d×n), one-hot labels (K×n), and integer labels |
| `NormalizeData` | Per-dimension normalization using provided mean and std |
| `InitNetwork` | Initializes W and b with Gaussian noise (σ=0.01) |
| `softmax` | Numerically stable softmax (subtracts row max before exp) |
| `ApplyNetwork` | Forward pass: computes class probabilities P = softmax(Wx + b) |
| `ComputeLoss` | Mean cross-entropy loss |
| `ComputeCost` | Loss + λ‖W‖² regularization term |
| `ComputeAccuracy` | Classification accuracy (argmax prediction vs ground truth) |
| `BackwardPass` | Analytic gradient of cost w.r.t. W and b |
| `ComputeGradsWithTorch` | Reference gradients via PyTorch autograd |
| `MiniBatchGD` | Full training loop with per-epoch logging |
| `PlotHistory` | Plots train/val loss and cost curves |
| `VisualizeWeights` | Reshapes and displays W rows as 32×32×3 class template images |

## Results

### Gradient Verification

Analytic gradients match PyTorch to machine precision:

| λ | Max Abs Error (W) | Max Rel Error (W) |
|---|---|---|
| 0.0 | 1.11e-16 | 2.10e-15 |
| 0.1 | 1.11e-16 | 4.58e-15 |

### Training Configurations

All configs use `n_batch=100`, `n_epochs=40`:

| λ | η | Train Acc | Val Acc | Test Acc |
|---|---|---|---|---|
| 0.0 | 0.1 | 42.12% | 27.10% | 27.70% |
| 0.0 | 0.001 | 45.57% | 38.46% | 39.21% |
| 0.1 | 0.001 | 44.63% | 38.62% | 39.30% |
| 1.0 | 0.001 | 39.85% | 36.32% | 37.55% |

### Key Findings

- **Learning rate matters:** η=0.1 causes wild oscillations and poor generalization (27.70% test). η=0.001 gives stable convergence (39.21% test).
- **Moderate regularization helps slightly:** λ=0.1 achieves the best test accuracy (39.30%) and reduces the train/val gap.
- **Strong regularization causes underfitting:** λ=1.0 nearly eliminates overfitting but plateaus early, dropping test accuracy to 37.55%. However, it produces the most visually interpretable weight templates.

## License

Academic coursework, DD2424 Deep Learning in Data Science, KTH Royal Institute of Technology.