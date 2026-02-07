"""Utilities for loading pre-trained black-box classifiers.

This module provides functions to load and freeze pre-trained classifiers
for MNIST and Adult datasets. Classifiers are loaded from checkpoints and
set to evaluation mode with all parameters frozen.
"""

import os
import sys
import io
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

from .mnist_cnn import create_mnist_classifier, MNISTCNNAdvanced
from .adult_mlp import create_adult_classifier, AdultMLPAdvanced


class ModuleRemapper(pickle.Unpickler):
    """Custom unpickler to handle module path remapping.

    This is needed because some checkpoints were saved with old module paths
    that need to be remapped to the new package structure.
    """

    def find_class(self, module, name):
        # Remap old module paths to new ones
        if module.startswith('probe_cfs'):
            # Remap probe_cfs.clf_models to cf_tcvae.classifiers
            if 'adult_mlp' in module:
                module = 'cf_tcvae.classifiers.adult_mlp'
            elif 'mnist_cnn' in module:
                module = 'cf_tcvae.classifiers.mnist_cnn'
            else:
                # For other probe_cfs modules, try to find in current package
                pass

        return super().find_class(module, name)


def get_checkpoint_path(dataset: str, package_dir: Optional[str] = None) -> str:
    """Get the path to classifier checkpoint.

    Args:
        dataset: Dataset name ('mnist' or 'adult')
        package_dir: Optional package directory path. If None, uses relative path.

    Returns:
        Absolute path to checkpoint file

    Raises:
        ValueError: If dataset is not recognized
        FileNotFoundError: If checkpoint file does not exist
    """
    if dataset.lower() == 'mnist':
        checkpoint_name = 'mnist_advanced_best.pth'
    elif dataset.lower() == 'adult':
        checkpoint_name = 'adult_mlp_full.pth'
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'mnist' or 'adult'")

    # Determine checkpoint directory
    if package_dir is None:
        # Use relative path from this file
        current_dir = Path(__file__).parent
        checkpoint_dir = current_dir / 'checkpoints'
    else:
        checkpoint_dir = Path(package_dir) / 'cf_tcvae' / 'classifiers' / 'checkpoints'

    checkpoint_path = checkpoint_dir / checkpoint_name

    # CRITICAL: No fallbacks - raise error if file not found
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found at {checkpoint_path}. "
            f"Expected checkpoint for dataset '{dataset}' at this location. "
            f"Please ensure checkpoints are properly installed."
        )

    return str(checkpoint_path)


def load_mnist_classifier(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    freeze: bool = True
) -> Tuple[nn.Module, dict]:
    """Load pre-trained MNIST CNN classifier.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses default location.
        device: Device to load model on. If None, uses CUDA if available.
        freeze: Whether to freeze all parameters (default: True)

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains
        training metadata and accuracy information

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If checkpoint loading fails
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path('mnist')

    # Create model architecture
    model = MNISTCNNAdvanced(num_classes=10, dropout_rate=0.2)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    # Move to device
    model = model.to(device)

    # Set to eval mode
    model.eval()

    # Freeze parameters if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, checkpoint


def load_adult_classifier(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    freeze: bool = True,
    input_dim: int = 108
) -> Tuple[nn.Module, dict]:
    """Load pre-trained Adult MLP classifier.

    NOTE: The original Adult classifier checkpoint has module dependencies from
    the original codebase ('probe_cfs'). This is a known issue documented in
    plan_progress.md Step 1. The checkpoint loads correctly in the original
    environment but requires retraining for the standalone repository.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses default location.
        device: Device to load model on. If None, uses CUDA if available.
        freeze: Whether to freeze all parameters (default: True)
        input_dim: Input feature dimension (default: 108)

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains
        training metadata and accuracy information

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If checkpoint loading fails (including module dependency issues)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path('adult')

    # Create model architecture
    model = AdultMLPAdvanced(input_dim=input_dim, num_classes=2, dropout_rate=0.2)

    # Load checkpoint with custom unpickler to handle module remapping
    try:
        # Try standard loading first
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except ModuleNotFoundError as e:
        # If we get a module not found error, try with custom unpickler
        try:
            with open(checkpoint_path, 'rb') as f:
                unpickler = ModuleRemapper(f)
                checkpoint = unpickler.load()
        except Exception as inner_e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}. "
                f"Standard load error: {e}. Custom unpickler error: {inner_e}"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    # Move to device
    model = model.to(device)

    # Set to eval mode
    model.eval()

    # Freeze parameters if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, checkpoint


def load_classifier(
    dataset: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    freeze: bool = True,
    **kwargs
) -> Tuple[nn.Module, dict]:
    """Load pre-trained classifier for specified dataset.

    Args:
        dataset: Dataset name ('mnist' or 'adult')
        checkpoint_path: Path to checkpoint. If None, uses default location.
        device: Device to load model on. If None, uses CUDA if available.
        freeze: Whether to freeze all parameters (default: True)
        **kwargs: Additional dataset-specific arguments

    Returns:
        Tuple of (model, checkpoint_dict)

    Raises:
        ValueError: If dataset is not recognized
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If checkpoint loading fails
    """
    if dataset.lower() == 'mnist':
        return load_mnist_classifier(checkpoint_path, device, freeze)
    elif dataset.lower() == 'adult':
        return load_adult_classifier(checkpoint_path, device, freeze, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'mnist' or 'adult'")


def verify_frozen(model: nn.Module) -> bool:
    """Verify that all model parameters are frozen.

    Args:
        model: PyTorch model to check

    Returns:
        True if all parameters have requires_grad=False, False otherwise
    """
    return all(not param.requires_grad for param in model.parameters())


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module) -> dict:
    """Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing model information
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen = verify_frozen(model)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'all_frozen': frozen,
        'device': next(model.parameters()).device,
        'dtype': next(model.parameters()).dtype
    }
