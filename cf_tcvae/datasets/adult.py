"""
Adult Income dataset loader for CF-TCVAE.

This module provides PyTorch Dataset and DataLoader utilities for the Adult Income
dataset, including preprocessing with one-hot encoding for categorical features.

The Adult dataset preprocessing follows these steps:
1. Load data from adult.data and adult.test files
2. Handle missing values by dropping rows
3. Encode categorical features using one-hot encoding
4. Standardize continuous features using StandardScaler
5. Split into train/val/test sets

Preprocessing metadata (scaler and label encoders) is loaded from a pickle file
to ensure consistency with the pre-trained black-box classifier.
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class AdultDataset(Dataset):
    """
    PyTorch Dataset for Adult Income data.

    Args:
        X: Feature array of shape [n_samples, 108]
        y: Label array of shape [n_samples]
    """

    def __init__(self, X, y):
        # Convert to numpy arrays if pandas Series/DataFrame
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Validate shapes
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

        # Validate feature dimension
        if self.X.shape[1] != 108:
            raise ValueError(
                f"Adult dataset must have 108 features after preprocessing. "
                f"Got {self.X.shape[1]} features. Check preprocessing_metadata.pkl."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AdultPairsDataset(Dataset):
    """
    Dataset for counterfactual generation with (x, y_orig, y_target) pairs.

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


class AdultDataModule:
    """
    Adult Income dataset loader with train/val/test splits.

    The Adult dataset is a binary classification task predicting whether income
    exceeds $50K/year based on census data. After preprocessing:
    - 6 continuous features (standardized)
    - 8 categorical features (one-hot encoded)
    - Total: 108 features
    - 2 classes: <=50K (0), >50K (1)

    Args:
        root: Root directory containing adult.data, adult.test, and preprocessing_metadata.pkl
        batch_size: Batch size for training
        test_batch_size: Batch size for validation/test
        val_split: Fraction of training data for validation (default: 0.1)

    Attributes:
        input_dim: Number of features after preprocessing (108)
        n_classes: Number of target classes (2)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """

    def __init__(self, root='./cf_tcvae/datasets/data/adult', batch_size=1024,
                 test_batch_size=1024, val_split=0.1):
        self.root = root
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split

        # Adult dataset attributes
        self.input_dim = 108  # Fixed after preprocessing
        self.n_classes = 2  # Binary classification

        # Feature definitions
        self.continuous_features = [
            'age', 'fnlwgt', 'education-num',
            'capital-gain', 'capital-loss', 'hours-per-week'
        ]
        self.categorical_features = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country'
        ]

        # Will be loaded from preprocessing_metadata.pkl
        self.scaler = None
        self.label_encoders = None

        # Load and preprocess data
        self._load_and_preprocess()

    def _load_preprocessing_metadata(self):
        """Load preprocessing metadata (scaler and label encoders)."""
        metadata_path = os.path.join(self.root, 'preprocessing_metadata.pkl')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Preprocessing metadata not found at {metadata_path}. "
                f"This file is required to ensure consistency with the pre-trained classifier. "
                f"Expected structure: {self.root}/preprocessing_metadata.pkl"
            )

        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessing metadata: {e}")

        # Validate metadata
        if 'scaler' not in metadata or 'label_encoders' not in metadata:
            raise ValueError(
                f"Invalid preprocessing metadata. Expected keys: 'scaler', 'label_encoders'. "
                f"Got: {list(metadata.keys())}"
            )

        self.scaler = metadata['scaler']
        self.label_encoders = metadata['label_encoders']

        print(f"Loaded preprocessing metadata from {metadata_path}")
        print(f"  Scaler: {type(self.scaler).__name__}")
        print(f"  Label encoders: {len(self.label_encoders)} categorical features")

    def _load_and_preprocess(self):
        """Load and preprocess the Adult dataset."""
        # Check if data files exist
        train_path = os.path.join(self.root, 'adult.data')
        test_path = os.path.join(self.root, 'adult.test')

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Adult training data not found at {train_path}. "
                f"Expected structure: {self.root}/adult.data"
            )
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Adult test data not found at {test_path}. "
                f"Expected structure: {self.root}/adult.test"
            )

        # Load preprocessing metadata
        self._load_preprocessing_metadata()

        # Column names for Adult dataset
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]

        # Load data
        try:
            train_data = pd.read_csv(train_path, names=columns, na_values=' ?',
                                    skipinitialspace=True)
            # Test file has a header, so skip first row
            test_data = pd.read_csv(test_path, names=columns, na_values=' ?',
                                   skipinitialspace=True, skiprows=1)
        except Exception as e:
            raise RuntimeError(f"Failed to load Adult data files: {e}")

        # Clean income column in test set (remove trailing period)
        test_data['income'] = test_data['income'].str.rstrip('.')

        # Combine for consistent preprocessing
        all_data = pd.concat([train_data, test_data], ignore_index=True)

        # Handle missing values by dropping rows
        initial_count = len(all_data)
        all_data = all_data.dropna()
        dropped_count = initial_count - len(all_data)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} rows with missing values")

        # Separate features and target
        X = all_data.drop('income', axis=1)
        y = all_data['income']

        # Encode target: <=50K -> 0, >50K -> 1
        y = (y == '>50K').astype(int).values

        # Process categorical features using loaded encoders
        categorical_data = []
        for feature in self.categorical_features:
            if feature not in self.label_encoders:
                raise ValueError(
                    f"Label encoder for feature '{feature}' not found in preprocessing metadata"
                )

            try:
                encoded = self.label_encoders[feature].transform(X[feature])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to encode categorical feature '{feature}': {e}. "
                    f"Check that the data matches the preprocessing metadata."
                )

            n_categories = len(self.label_encoders[feature].classes_)
            one_hot = np.eye(n_categories)[encoded]
            categorical_data.append(one_hot)

        # Concatenate categorical features
        if categorical_data:
            categorical_features = np.concatenate(categorical_data, axis=1)
        else:
            categorical_features = np.empty((len(X), 0))

        # Process continuous features using loaded scaler
        continuous_data = X[self.continuous_features].values.astype(np.float32)

        try:
            continuous_data = self.scaler.transform(continuous_data)
        except Exception as e:
            raise RuntimeError(f"Failed to scale continuous features: {e}")

        # Combine continuous and categorical features
        X_processed = np.concatenate([continuous_data, categorical_features], axis=1)

        # Validate feature dimension
        if X_processed.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features after preprocessing, "
                f"got {X_processed.shape[1]}. Check preprocessing metadata."
            )

        # Split back into train and test
        n_train = len(train_data.dropna())
        X_train = X_processed[:n_train]
        y_train = y[:n_train]
        X_test = X_processed[n_train:]
        y_test = y[n_train:]

        # Create train/val split from training data
        if self.val_split <= 0 or self.val_split >= 1:
            raise ValueError(f"val_split must be in (0, 1). Got {self.val_split}")

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_split,
                random_state=42, stratify=y_train
            )
        except Exception as e:
            raise RuntimeError(f"Failed to split training data: {e}")

        # Create datasets
        self.train_dataset = AdultDataset(X_train, y_train)
        self.val_dataset = AdultDataset(X_val, y_val)
        self.test_dataset = AdultDataset(X_test, y_test)

        print(f"Adult dataset loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Feature dimension: {self.input_dim}")
        print(f"  Classes: {self.n_classes} (0: <=50K, 1: >50K)")

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


def create_adult_loaders(root='./cf_tcvae/datasets/data/adult', batch_size=1024,
                         test_batch_size=1024, val_split=0.1, num_workers=4):
    """
    Convenience function to create Adult dataset loaders.

    Args:
        root: Root directory containing Adult data files
        batch_size: Training batch size
        test_batch_size: Validation/test batch size
        val_split: Fraction of training data for validation
        num_workers: Number of data loading workers

    Returns:
        Tuple of (data_module, train_loader, val_loader, test_loader)

    Example:
        >>> data_module, train_loader, val_loader, test_loader = create_adult_loaders()
        >>> print(f"Feature dimension: {data_module.input_dim}")  # 108
        >>> for batch_idx, (features, labels) in enumerate(train_loader):
        >>>     # features shape: [batch_size, 108]
        >>>     # labels shape: [batch_size]
        >>>     pass
    """
    data_module = AdultDataModule(root, batch_size, test_batch_size, val_split)

    train_loader = data_module.train_dataloader(num_workers=num_workers)
    val_loader = data_module.val_dataloader(num_workers=num_workers)
    test_loader = data_module.test_dataloader(num_workers=num_workers)

    return data_module, train_loader, val_loader, test_loader
