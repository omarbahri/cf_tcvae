"""
Dataset loaders for CF-TCVAE.

This module provides dataset loading utilities for MNIST and Adult Income datasets,
with proper preprocessing and train/val/test splits.

Available datasets:
- MNIST: 28x28 grayscale images, 10 classes (digits 0-9)
- Adult Income: Tabular data, 108 features, 2 classes (<=50K, >50K)
"""

from .mnist import (
    MNISTDataset,
    MNISTPairsDataset,
    create_mnist_loaders
)

from .adult import (
    AdultDataModule,
    AdultDataset,
    AdultPairsDataset,
    create_adult_loaders
)

__all__ = [
    # MNIST
    'MNISTDataset',
    'MNISTPairsDataset',
    'create_mnist_loaders',
    # Adult
    'AdultDataModule',
    'AdultDataset',
    'AdultPairsDataset',
    'create_adult_loaders',
]
