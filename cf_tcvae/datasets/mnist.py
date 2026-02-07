"""
MNIST dataset loader for CF-TCVAE.

This module provides PyTorch Dataset and DataLoader utilities for MNIST,
supporting both standard training and counterfactual generation workflows.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataset:
    """
    MNIST dataset loader with train/val/test splits.

    Args:
        root: Root directory containing MNIST data files
        batch_size: Batch size for training
        test_batch_size: Batch size for validation/test
        val_split: Fraction of training data to use for validation (default: 5000 samples)

    Attributes:
        dims: Image dimensions (1, 28, 28)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """

    def __init__(self, root='./cf_tcvae/datasets/data', batch_size=1024,
                 test_batch_size=1024, val_samples=5000):
        self.root = root
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_samples = val_samples

        # MNIST dimensions
        self.dims = (1, 28, 28)

        # Simple transforms: convert to tensor (values in [0, 1])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load datasets
        self._load_datasets()

    def _load_datasets(self):
        """Load and split MNIST datasets."""
        # Check if data directory exists
        mnist_dir = os.path.join(self.root, 'MNIST')
        if not os.path.exists(mnist_dir):
            raise FileNotFoundError(
                f"MNIST data directory not found at {mnist_dir}. "
                f"Expected structure: {self.root}/MNIST/raw/train-images-idx3-ubyte, etc."
            )

        # Load training data
        try:
            mnist_full = datasets.MNIST(self.root, train=True, transform=self.transforms,
                                       download=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MNIST training data from {self.root}. "
                f"Ensure raw files exist in {mnist_dir}/raw/. Error: {e}"
            )

        # Split into train and validation with fixed seed for reproducibility
        train_samples = len(mnist_full) - self.val_samples
        if train_samples <= 0 or self.val_samples <= 0:
            raise ValueError(
                f"Invalid split: train_samples={train_samples}, val_samples={self.val_samples}. "
                f"Total samples: {len(mnist_full)}"
            )

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            mnist_full, [train_samples, self.val_samples], generator=generator
        )

        # Load test data
        try:
            self.test_dataset = datasets.MNIST(self.root, train=False,
                                              transform=self.transforms, download=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MNIST test data from {self.root}. Error: {e}"
            )

        print(f"MNIST dataset loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, shuffle=True, num_workers=4):
        """
        Create training DataLoader.

        Args:
            shuffle: Whether to shuffle training data (default: True)
            num_workers: Number of data loading workers (default: 4)

        Returns:
            DataLoader for training set
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def val_dataloader(self, num_workers=4):
        """
        Create validation DataLoader.

        Args:
            num_workers: Number of data loading workers (default: 4)

        Returns:
            DataLoader for validation set
        """
        def worker_init_fn(worker_id):
            """Ensure reproducible validation batches."""
            np.random.seed(42 + worker_id)
            torch.manual_seed(42 + worker_id)

        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

    def test_dataloader(self, num_workers=4):
        """
        Create test DataLoader.

        Args:
            num_workers: Number of data loading workers (default: 4)

        Returns:
            DataLoader for test set
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


class MNISTPairsDataset(Dataset):
    """
    Dataset for counterfactual generation with (x, y_orig, y_target) pairs.

    This is used when training with all target classes mode, where each sample
    is paired with all possible target classes.

    Args:
        pairs: List of tuples (x, y_orig, y_target)
    """

    def __init__(self, pairs):
        if pairs is None or len(pairs) == 0:
            raise ValueError("pairs cannot be None or empty")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y_orig, y_target = self.pairs[idx]
        return x, y_orig, y_target


def create_mnist_loaders(root='./cf_tcvae/datasets/data', batch_size=1024,
                         test_batch_size=1024, val_samples=5000, num_workers=4):
    """
    Convenience function to create MNIST data loaders.

    Args:
        root: Root directory containing MNIST data
        batch_size: Training batch size
        test_batch_size: Validation/test batch size
        val_samples: Number of validation samples (default: 5000)
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_mnist_loaders()
        >>> for batch_idx, (images, labels) in enumerate(train_loader):
        >>>     # images shape: [batch_size, 1, 28, 28]
        >>>     # labels shape: [batch_size]
        >>>     pass
    """
    dataset = MNISTDataset(root, batch_size, test_batch_size, val_samples)

    train_loader = dataset.train_dataloader(num_workers=num_workers)
    val_loader = dataset.val_dataloader(num_workers=num_workers)
    test_loader = dataset.test_dataloader(num_workers=num_workers)

    return train_loader, val_loader, test_loader
