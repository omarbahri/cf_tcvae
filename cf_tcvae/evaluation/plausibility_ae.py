"""
Plausibility metrics using autoencoders.

Implements:
- Plausibility (AE): Global autoencoder trained on all data
- Plausibility (tc-AE): Per-class autoencoders trained on class-specific data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ==================== AUTOENCODER ARCHITECTURES ====================

class MNISTAutoencoder(nn.Module):
    """
    Convolutional autoencoder for MNIST images.

    Architecture matches the VAE encoder/decoder but uses MSE reconstruction loss.
    """

    def __init__(self):
        super(MNISTAutoencoder, self).__init__()

        # Encoder: similar to MNIST CNN encoder but simpler
        self.encoder = nn.Sequential(
            # 28x28x1 -> 14x14x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 14x14x32 -> 7x7x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 7x7x64 -> 4x4x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Flatten: 4x4x128 = 2048
            nn.Flatten(),

            # FC to latent: 2048 -> 128
            nn.Linear(4 * 4 * 128, 128),
            nn.ReLU(inplace=True)
        )

        # Decoder: mirror of encoder
        self.decoder = nn.Sequential(
            # 128 -> 2048
            nn.Linear(128, 4 * 4 * 128),
            nn.ReLU(inplace=True),

            # Reshape: 2048 -> 4x4x128
            nn.Unflatten(1, (128, 4, 4)),

            # 4x4x128 -> 7x7x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 7x7x64 -> 14x14x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 14x14x32 -> 28x28x1
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TabularAutoencoder(nn.Module):
    """
    MLP autoencoder for tabular data (Adult dataset).

    Architecture matches the VAE encoder/decoder but uses MSE reconstruction loss.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super(TabularAutoencoder, self).__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: similar to Adult MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder: mirror of encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output in [0, 1] for normalized features
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==================== TRAINING FUNCTIONS ====================

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    likelihood: str = 'bernoulli',
    epochs: int = 50,
    lr: float = 1e-3,
    verbose: bool = True
) -> nn.Module:
    """
    Train an autoencoder on the provided data.

    Args:
        model: Autoencoder model (MNISTAutoencoder or TabularAutoencoder)
        train_loader: DataLoader with training data
        device: Device for training
        likelihood: Loss type ('bernoulli' for BCE, 'gaussian' for MSE)
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress

    Returns:
        Trained model

    Raises:
        ValueError: If likelihood is invalid
    """
    if likelihood not in ['bernoulli', 'gaussian']:
        raise ValueError(f"likelihood must be 'bernoulli' or 'gaussian', got '{likelihood}'")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Choose loss function
    if likelihood == 'bernoulli':
        criterion = nn.BCELoss()
    else:  # gaussian
        criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            # Handle both (x, y) and (x,) formats
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0]
            else:
                x = batch_data

            x = x.to(device)

            # Forward pass
            reconstructed = model(x)
            loss = criterion(reconstructed, x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    model.eval()

    if verbose:
        logger.info(f"Autoencoder training completed. Final loss: {avg_loss:.6f}")

    return model


def train_global_autoencoder(
    dataset: torch.utils.data.Dataset,
    data_type: str = 'image',
    input_dim: Optional[int] = None,
    latent_dim: int = 128,
    likelihood: str = 'bernoulli',
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Train a global autoencoder on the entire dataset.

    Args:
        dataset: Training dataset (returns (x, y) or x)
        data_type: 'image' for MNIST, 'tabular' for Adult
        input_dim: Input dimension (required for tabular)
        latent_dim: Latent dimension (for tabular only)
        likelihood: 'bernoulli' or 'gaussian'
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        device: Device for training
        verbose: Whether to print progress

    Returns:
        Trained autoencoder

    Raises:
        ValueError: If data_type is invalid or input_dim is missing for tabular
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_type == 'image':
        model = MNISTAutoencoder()
    elif data_type == 'tabular':
        if input_dim is None:
            raise ValueError("input_dim is required for tabular autoencoders")
        model = TabularAutoencoder(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'image' or 'tabular'")

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if verbose:
        logger.info(f"Training global {data_type} autoencoder on {len(dataset)} samples...")

    # Train
    model = train_autoencoder(model, train_loader, device, likelihood, epochs, lr, verbose)

    return model


def train_class_autoencoders(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    data_type: str = 'image',
    input_dim: Optional[int] = None,
    latent_dim: int = 128,
    likelihood: str = 'bernoulli',
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[int, nn.Module]:
    """
    Train per-class autoencoders.

    Args:
        dataset: Training dataset that returns (x, y)
        num_classes: Number of classes
        data_type: 'image' for MNIST, 'tabular' for Adult
        input_dim: Input dimension (required for tabular)
        latent_dim: Latent dimension (for tabular only)
        likelihood: 'bernoulli' or 'gaussian'
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        device: Device for training
        verbose: Whether to print progress

    Returns:
        Dictionary mapping class label to trained autoencoder

    Raises:
        ValueError: If data_type is invalid or input_dim is missing for tabular
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect class indices
    class_indices = {c: [] for c in range(num_classes)}

    for idx, (x, y) in enumerate(dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        class_indices[y].append(idx)

    if verbose:
        logger.info(f"Training {num_classes} class-specific autoencoders...")
        for c, indices in class_indices.items():
            logger.info(f"  Class {c}: {len(indices)} samples")

    # Train one autoencoder per class
    class_autoencoders = {}

    for c in range(num_classes):
        indices = class_indices[c]

        if len(indices) == 0:
            logger.warning(f"No samples for class {c}, skipping")
            continue

        # Create subset for this class
        class_subset = Subset(dataset, indices)

        # Create model
        if data_type == 'image':
            model = MNISTAutoencoder()
        elif data_type == 'tabular':
            if input_dim is None:
                raise ValueError("input_dim is required for tabular autoencoders")
            model = TabularAutoencoder(input_dim, latent_dim)
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'image' or 'tabular'")

        # Create data loader
        class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=True, num_workers=0)

        if verbose:
            logger.info(f"Training autoencoder for class {c}...")

        # Train
        model = train_autoencoder(model, class_loader, device, likelihood, epochs, lr, verbose=False)
        class_autoencoders[c] = model

        if verbose:
            logger.info(f"  Class {c} autoencoder trained")

    if verbose:
        logger.info(f"Trained {len(class_autoencoders)}/{num_classes} class autoencoders")

    return class_autoencoders


# ==================== PLAUSIBILITY COMPUTATION ====================

def compute_plausibility_ae(
    counterfactuals: torch.Tensor,
    autoencoder: nn.Module,
    likelihood: str = 'bernoulli'
) -> float:
    """
    Compute Plausibility (AE): reconstruction error using global autoencoder.

    Plausibility(AE) = (1/N·d) Σ_i Σ_j ℓ(AE(x_cf)_ij, x_cf_ij)

    Args:
        counterfactuals: Generated counterfactuals [N, *input_shape]
        autoencoder: Trained global autoencoder
        likelihood: 'bernoulli' for BCE, 'gaussian' for MSE

    Returns:
        Mean reconstruction loss per feature

    Raises:
        ValueError: If inputs are invalid
    """
    if counterfactuals is None or autoencoder is None:
        raise ValueError("counterfactuals and autoencoder are required")

    if likelihood not in ['bernoulli', 'gaussian']:
        raise ValueError(f"likelihood must be 'bernoulli' or 'gaussian', got '{likelihood}'")

    autoencoder.eval()
    device = next(autoencoder.parameters()).device
    counterfactuals = counterfactuals.to(device)

    with torch.no_grad():
        reconstructed = autoencoder(counterfactuals)

        # Compute reconstruction loss per sample per feature
        if likelihood == 'bernoulli':
            # BCE loss
            loss = F.binary_cross_entropy(reconstructed, counterfactuals, reduction='none')
        else:  # gaussian
            # MSE loss
            loss = F.mse_loss(reconstructed, counterfactuals, reduction='none')

        # Average over all features and all samples
        mean_loss = loss.mean().item()

    return mean_loss


def compute_plausibility_tc_ae(
    counterfactuals: torch.Tensor,
    target_classes: torch.Tensor,
    class_autoencoders: Dict[int, nn.Module],
    likelihood: str = 'bernoulli'
) -> float:
    """
    Compute Plausibility (tc-AE): reconstruction error using per-class autoencoders.

    Plausibility(tc-AE) = (1/N·d) Σ_i Σ_j ℓ(AE_{y'_i}(x_cf_i)_j, x_cf_i_j)

    For each counterfactual, use the autoencoder trained on its target class.

    Args:
        counterfactuals: Generated counterfactuals [N, *input_shape]
        target_classes: Target class labels [N]
        class_autoencoders: Dictionary mapping class label to trained autoencoder
        likelihood: 'bernoulli' for BCE, 'gaussian' for MSE

    Returns:
        Mean reconstruction loss per feature

    Raises:
        ValueError: If inputs are invalid
    """
    if counterfactuals is None or target_classes is None or class_autoencoders is None:
        raise ValueError("counterfactuals, target_classes, and class_autoencoders are required")

    if likelihood not in ['bernoulli', 'gaussian']:
        raise ValueError(f"likelihood must be 'bernoulli' or 'gaussian', got '{likelihood}'")

    if len(class_autoencoders) == 0:
        raise ValueError("class_autoencoders is empty")

    # Get device from first autoencoder
    device = next(iter(class_autoencoders.values())).parameters().__next__().device
    counterfactuals = counterfactuals.to(device)
    target_classes = target_classes.to(device)

    all_losses = []

    for c in torch.unique(target_classes):
        c_int = c.item()

        if c_int not in class_autoencoders:
            logger.warning(f"No autoencoder for class {c_int}, skipping")
            continue

        # Get counterfactuals for this class
        mask = (target_classes == c)
        cf_c = counterfactuals[mask]

        if len(cf_c) == 0:
            continue

        # Get autoencoder for this class
        ae_c = class_autoencoders[c_int]
        ae_c.eval()
        ae_c = ae_c.to(device)

        with torch.no_grad():
            reconstructed = ae_c(cf_c)

            # Compute reconstruction loss
            if likelihood == 'bernoulli':
                loss = F.binary_cross_entropy(reconstructed, cf_c, reduction='none')
            else:  # gaussian
                loss = F.mse_loss(reconstructed, cf_c, reduction='none')

            # Mean over features for this class
            all_losses.append(loss.mean().item())

    if len(all_losses) == 0:
        raise ValueError("No valid class autoencoders found for any target class")

    # Average across all classes
    mean_loss = float(np.mean(all_losses))

    return mean_loss


# ==================== SAVE/LOAD UTILITIES ====================

def save_autoencoder(
    model: nn.Module,
    path: str,
    data_type: str,
    class_label: Optional[int] = None,
    **kwargs
):
    """
    Save autoencoder checkpoint.

    Args:
        model: Autoencoder model
        path: Path to save checkpoint
        data_type: 'image' or 'tabular'
        class_label: Class label if this is a class-specific autoencoder
        **kwargs: Additional metadata to save
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'state_dict': model.state_dict(),
        'data_type': data_type,
        'class_label': class_label,
        **kwargs
    }

    torch.save(checkpoint, path)
    logger.info(f"Autoencoder saved to {path}")


def load_autoencoder(
    path: str,
    data_type: str,
    input_dim: Optional[int] = None,
    latent_dim: int = 128,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load autoencoder checkpoint.

    Args:
        path: Path to checkpoint
        data_type: 'image' or 'tabular'
        input_dim: Input dimension (required for tabular)
        latent_dim: Latent dimension (for tabular)
        device: Device to load model on

    Returns:
        Loaded autoencoder model

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If data_type is invalid or input_dim is missing
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    if data_type == 'image':
        model = MNISTAutoencoder()
    elif data_type == 'tabular':
        if input_dim is None:
            raise ValueError("input_dim is required for tabular autoencoders")
        model = TabularAutoencoder(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'image' or 'tabular'")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Autoencoder loaded from {path}")

    return model
