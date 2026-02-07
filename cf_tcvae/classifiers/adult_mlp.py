"""Adult Dataset MLP Classifier Implementation.

This module implements MLP architectures for Adult Income binary classification.
The models are designed for tabular data with one-hot encoded categorical features.

Architecture:
- Fully connected layers with batch normalization
- Residual connections for better gradient flow
- Dropout regularization
- Optimized for 108-dimensional preprocessed Adult features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ResidualMLPBlock(nn.Module):
    """Residual block for MLP with improved gradient flow."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super(ResidualMLPBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # Residual connection
        out = F.relu(out)
        return out


class AdultMLPBasic(nn.Module):
    """Basic MLP for Adult dataset binary classification."""

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dims: Optional[List[int]] = None,
                 dropout_rate: float = 0.3):
        super(AdultMLPBasic, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Build the network
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AdultMLPAdvanced(nn.Module):
    """Advanced MLP with residual connections for Adult dataset.

    This architecture is used for the pre-trained black-box classifier.
    """

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dims: Optional[List[int]] = None,
                 dropout_rate: float = 0.2, num_residual_blocks: int = 2):
        super(AdultMLPAdvanced, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dims[0], dropout_rate)
            for _ in range(num_residual_blocks)
        ])

        # Hidden layers
        layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Hidden layers
        x = self.hidden_layers(x)

        # Classification
        x = self.classifier(x)

        return x


class AdultMLPEfficient(nn.Module):
    """Efficient MLP with fewer parameters for Adult dataset."""

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dims: Optional[List[int]] = None,
                 dropout_rate: float = 0.25, width_multiplier: float = 1.0):
        super(AdultMLPEfficient, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Default hidden dimensions scaled by width multiplier
        if hidden_dims is None:
            base_dims = [128, 64]
            hidden_dims = [int(dim * width_multiplier) for dim in base_dims]

        # Build the network
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Middle layers
        self.middle_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.middle_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))

        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        # Store dimensions
        self.hidden_dims = hidden_dims

    def forward(self, x):
        x = self.input_layer(x)

        # Process through middle layers
        for layer in self.middle_layers:
            x = layer(x)

        # Classification
        x = self.classifier(x)

        return x


def create_adult_classifier(model_type: str = "advanced", input_dim: int = 108, **kwargs) -> nn.Module:
    """Factory function to create Adult MLP classifiers.

    Args:
        model_type: Type of model ('basic', 'advanced', 'efficient')
        input_dim: Input feature dimension (default: 108 for preprocessed Adult)
        **kwargs: Additional arguments for model initialization

    Returns:
        PyTorch MLP model for Adult binary classification

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "basic":
        return AdultMLPBasic(input_dim=input_dim, **kwargs)
    elif model_type == "advanced":
        return AdultMLPAdvanced(input_dim=input_dim, **kwargs)
    elif model_type == "efficient":
        return AdultMLPEfficient(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'basic', 'advanced', 'efficient'")
