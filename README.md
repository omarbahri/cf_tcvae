# CF-TCVAE: Robust Counterfactual Explanations via Class-Conditional β-TCVAE

Official implementation for the IJCAI 2026 paper "Robust Counterfactual Explanations via Class-Conditional β-TCVAE".

## Overview

CF-TCVAE is a deep generative model for creating robust counterfactual explanations. It combines class-conditional variational autoencoders with total correlation (TC) regularization to generate counterfactuals that are:

- **Valid**: Successfully classified as the target class
- **Proximate**: Close to the original input
- **Plausible**: Realistic and in-distribution for the target class
- **Robust**: Stable under input perturbations

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone or navigate to the repository:
```bash
cd cf_tcvae
```

2. Activate the conda environment:
```bash
conda activate robust-cfvae
```

3. Install the package in development mode:
```bash
pip install -e .
```

The `robust-cfvae` conda environment should already have all necessary dependencies installed (PyTorch, NumPy, scikit-learn, etc.). If you need to install from scratch, use:
```bash
pip install -r requirements.txt
```

## Repository Structure

```
cf_tcvae/
├── cf_tcvae/                      # Main package
│   ├── models/                    # Model architectures
│   │   ├── cf_tcvae.py           # Main CF-TCVAE model
│   │   ├── encoders.py           # Encoder architectures
│   │   ├── decoders.py           # Decoder architectures
│   │   ├── tc_estimator.py       # TC estimation
│   │   ├── priors.py             # Class-conditional priors
│   │   └── auxiliary.py          # Auxiliary classifier
│   ├── datasets/                  # Dataset loaders
│   │   ├── mnist.py              # MNIST loader
│   │   ├── adult.py              # Adult Income loader
│   │   └── data/                 # Dataset files
│   ├── classifiers/               # Pre-trained black-box classifiers
│   │   ├── mnist_cnn.py          # MNIST CNN
│   │   ├── adult_mlp.py          # Adult MLP
│   │   └── checkpoints/          # Pre-trained weights
│   ├── training/                  # Training utilities
│   │   ├── trainer.py            # Training loop
│   │   └── losses.py             # Loss functions
│   └── evaluation/                # Evaluation metrics
│       ├── metrics.py            # All 8 metrics from paper
│       ├── validity.py           # Validity metrics
│       ├── proximity.py          # Proximity metrics
│       ├── plausibility.py       # Plausibility metrics
│       └── robustness.py         # Robustness metrics
├── scripts/                       # Training and evaluation scripts
│   ├── train_mnist.py            # MNIST training
│   ├── train_adult.py            # Adult training
│   ├── evaluate_mnist.py         # MNIST evaluation
│   └── evaluate_adult.py         # Adult evaluation
├── configs/                       # Configuration files
│   ├── mnist_default.yaml        # MNIST defaults
│   └── adult_default.yaml        # Adult defaults
└── tests/                         # Unit tests
```

## Datasets

The repository includes two benchmark datasets:

### MNIST
- **Task**: Digit classification (0-9)
- **Input**: 28x28 grayscale images
- **Instances**: 60,000 train, 10,000 test
- **Pre-trained classifier**: CNN with 99.7% test accuracy

### Adult Income
- **Task**: Income prediction (≤50K or >50K)
- **Input**: 108-dimensional one-hot encoded features
- **Instances**: 32,561 train, 16,281 test
- **Pre-trained classifier**: MLP with competitive accuracy

Both datasets and pre-trained classifiers are included in the repository.

## Quick Start

### Training CF-TCVAE

CF-TCVAE uses unified single-phase training (no progressive schedules or phase transitions). All loss components train jointly from the start.

**MNIST (with default parameters):**
```bash
python scripts/train_mnist.py \
    --max_epochs 150 \
    --batch_size 1024 \
    --beta 10.0 \
    --lr 0.001 \
    --epsilon 10.0 \
    --alpha_proximity 300.0 \
    --z_dim 64 \
    --save_dir checkpoints/mnist
```

**Adult Income (with default parameters):**
```bash
python scripts/train_adult.py \
    --max_epochs 200 \
    --batch_size 1024 \
    --beta 10.0 \
    --lr 0.005 \
    --epsilon 5.0 \
    --alpha_proximity 2.0 \
    --z_dim 16 \
    --save_dir checkpoints/adult
```

**Key Training Parameters:**
- `--beta`: TC regularization weight (default: 10.0 for both datasets)
- `--alpha_proximity`: Proximity loss weight (300.0 for MNIST, 2.0 for Adult)
- `--epsilon`: Hinge loss margin (10.0 for MNIST, 5.0 for Adult)
- `--lr`: Learning rate (0.001 for MNIST, 0.005 for Adult)
- `--z_dim`: Latent dimension (64 for MNIST, 16 for Adult)

All training scripts save logs to `logs/{script_name}/`.

### Evaluation

Evaluation computes all 8 metrics from Table 1 and saves results to CSV/JSON.

**MNIST:**
```bash
python scripts/evaluate_mnist.py \
    --checkpoint checkpoints/mnist/best_model.pth \
    --output_dir results/mnist \
    --save_visualizations \
    --num_visualizations 20

# Skip plausibility metrics to save time (no autoencoder training)
python scripts/evaluate_mnist.py \
    --checkpoint checkpoints/mnist/best_model.pth \
    --output_dir results/mnist \
    --skip_plausibility
```

**Adult Income:**
```bash
python scripts/evaluate_adult.py \
    --checkpoint checkpoints/adult/best_model.pth \
    --output_dir results/adult \
    --save_statistics

# Evaluate on subset (faster)
python scripts/evaluate_adult.py \
    --checkpoint checkpoints/adult/best_model.pth \
    --output_dir results/adult \
    --num_samples 1000 \
    --skip_plausibility
```

## Model Architecture

CF-TCVAE uses a class-conditional variational autoencoder with:

- **Encoder**: Maps input x and target class y' to latent distribution Q(z|x,y')
- **Decoder**: Generates counterfactual from z and y'
- **Class-conditional prior**: P(z|y') learned via EMA updates
- **TC regularization**: Disentangles latent factors within each class
- **Auxiliary classifier**: Improves class discrimination in latent space

### Loss Function

The unified training objective (Equation 15 from paper):

```
L = E_Q(z|x,y')[log P(x|z,y) + α·Dist(x,x_cf) + λ·HingeLoss(f(x_cf),y',ε)]
    + (β-1)·TC(z|y') + KL(Q(z|x,y')||P(z|y')) + L_aux(z,y)
```

Where:
- **Reconstruction**: -log P(x|z,y) encourages valid reconstruction
- **Proximity**: α·||x - x_cf||₁ keeps counterfactuals close
- **Validity**: λ·HingeLoss ensures target class prediction
- **TC regularization**: (β-1)·TC(z|y') disentangles latent factors
- **KL divergence**: Regularizes to class-conditional prior
- **Auxiliary loss**: Improves class discrimination

## Hyperparameters

### MNIST Defaults
- β = 10.0 (TC weight)
- lr = 0.001
- ε = 10.0 (hinge margin)
- α = 300.0 (proximity weight)
- z_dim = 64

### Adult Defaults
- β = 10.0 (TC weight)
- lr = 0.005
- ε = 5.0 (hinge margin)
- α = 2.0 (proximity weight)
- z_dim = 16

See `configs/` directory for complete default configurations.

## Set parameters in config files

You can specify all hyperparameters in YAML config files:

```bash
python scripts/train_mnist.py --config configs/mnist_default.yaml

# Override specific parameters
python scripts/train_mnist.py --config configs/mnist_default.yaml --beta 20.0 --lr 0.0005
```

Example config file (`configs/mnist_custom.yaml`):
```yaml
# Core CF-TCVAE parameters
beta: 10.0                    # TC regularization weight
lr: 0.001                     # Learning rate
epsilon: 10.0                 # Hinge loss margin
lambda_validity: 1.0          # Validity loss weight
alpha_proximity: 300.0        # Proximity loss weight

# Architecture
z_dim: 64                     # Latent dimension
num_ch: 32                    # CNN base channels
class_emb_dim: 32             # Class embedding dimension

# Training
batch_size: 1024
max_epochs: 150
lr_patience: 7                # ReduceLROnPlateau patience
lr_factor: 0.5                # LR reduction factor

# Prior
use_class_conditional_prior: true
prior_ema_momentum: 0.0001    # EMA momentum for prior updates

# Auxiliary classifier
use_aux_classifier: true
aux_weight: 0.5
aux_warmup_epochs: 10

# Device
device: cuda
seed: 42
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
