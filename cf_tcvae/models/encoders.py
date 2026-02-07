#!/usr/bin/env python3
"""Clean encoder architectures for CF-TCVAE (NO FiLM layers).

This module provides:
- CNN encoder for MNIST (image data)
- MLP encoder for Adult (tabular data)

All encoders produce (mu, logvar) for the latent distribution.
Class conditioning is handled via concatenation, NOT FiLM modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTEncoder(nn.Module):
    """CNN encoder for MNIST with class conditioning via concatenation.

    Architecture:
        Conv layers: 28x28 -> 14x14 -> 7x7 -> 4x4 -> 2x2
        Final: Flatten + Linear to produce (mu, logvar)

    Args:
        z_dim: Latent dimension
        num_ch: Number of base channels (default: 32)
        class_emb_dim: Dimension of class embeddings (0 for no conditioning)
    """

    def __init__(self, z_dim: int, num_ch: int = 32, class_emb_dim: int = 0):
        super().__init__()

        self.z_dim = z_dim
        self.num_ch = num_ch
        self.class_emb_dim = class_emb_dim

        # Convolutional layers (no activation in layer, added separately)
        self.conv1 = nn.Conv2d(1, num_ch, kernel_size=3, stride=2, padding=1)      # 28->14
        self.bn1 = nn.BatchNorm2d(num_ch)

        self.conv2 = nn.Conv2d(num_ch, num_ch*2, kernel_size=3, stride=2, padding=1)  # 14->7
        self.bn2 = nn.BatchNorm2d(num_ch*2)

        self.conv3 = nn.Conv2d(num_ch*2, num_ch*3, kernel_size=3, stride=2, padding=1)  # 7->4
        self.bn3 = nn.BatchNorm2d(num_ch*3)

        self.conv4 = nn.Conv2d(num_ch*3, num_ch*4, kernel_size=3, stride=2, padding=1)  # 4->2
        self.bn4 = nn.BatchNorm2d(num_ch*4)

        # Final conv to reduce spatial dimensions
        self.conv5 = nn.Conv2d(num_ch*4, z_dim//4, kernel_size=3, stride=1, padding=1)  # 2->2

        # Calculate flattened dimension
        base_output_dim = (z_dim // 4) * 2 * 2

        # Projection to latent with optional class conditioning
        if class_emb_dim > 0:
            # Concatenate conv output with class embedding
            self.mean_fc = nn.Linear(base_output_dim + class_emb_dim, z_dim)
            self.logvar_fc = nn.Linear(base_output_dim + class_emb_dim, z_dim)
        else:
            # No class conditioning
            self.mean_fc = nn.Linear(base_output_dim, z_dim)
            self.logvar_fc = nn.Linear(base_output_dim, z_dim)

        self._init_weights()

    def forward(self, x, class_emb=None):
        """
        Args:
            x: Input images [batch_size, 1, 28, 28]
            class_emb: Class embeddings [batch_size, class_emb_dim] or None

        Returns:
            z_mean: [batch_size, z_dim]
            z_logvar: [batch_size, z_dim]
        """
        # Ensure input shape
        if len(x.shape) == 2:
            # Assume flattened MNIST (784,) -> reshape to (1, 28, 28)
            x = x.view(-1, 1, 28, 28)

        # Convolutional layers with batch norm and ReLU
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.conv5(h)  # No activation on final conv

        # Flatten
        h = h.flatten(start_dim=1)

        # Concatenate with class embedding if provided
        if self.class_emb_dim > 0 and class_emb is not None:
            h = torch.cat([h, class_emb], dim=1)

        # Project to latent distribution parameters
        z_mean = self.mean_fc(h)
        z_logvar = self.logvar_fc(h)

        return z_mean, z_logvar

    def _init_weights(self):
        """Xavier initialization for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class AdultEncoder(nn.Module):
    """MLP encoder for Adult dataset with class conditioning via concatenation.

    Architecture:
        Input -> [256] -> [128] -> z_dim

    Args:
        z_dim: Latent dimension
        input_dim: Input feature dimension (108 for Adult)
        hidden_dims: List of hidden layer dimensions (default: [256, 128])
        class_emb_dim: Dimension of class embeddings (0 for no conditioning)
    """

    def __init__(self, z_dim: int, input_dim: int = 108,
                 hidden_dims=None, class_emb_dim: int = 0):
        super().__init__()

        self.z_dim = z_dim
        self.input_dim = input_dim
        self.class_emb_dim = class_emb_dim

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build hidden layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers)

        # Output heads with optional class conditioning
        final_dim = hidden_dims[-1]
        if class_emb_dim > 0:
            # Concatenate final hidden with class embedding
            self.mean_fc = nn.Linear(final_dim + class_emb_dim, z_dim)
            self.logvar_fc = nn.Linear(final_dim + class_emb_dim, z_dim)
        else:
            self.mean_fc = nn.Linear(final_dim, z_dim)
            self.logvar_fc = nn.Linear(final_dim, z_dim)

        self._init_weights()

    def forward(self, x, class_emb=None):
        """
        Args:
            x: Input features [batch_size, input_dim]
            class_emb: Class embeddings [batch_size, class_emb_dim] or None

        Returns:
            z_mean: [batch_size, z_dim]
            z_logvar: [batch_size, z_dim]
        """
        # Process through hidden layers
        h = self.hidden_layers(x)

        # Concatenate with class embedding if provided
        if self.class_emb_dim > 0 and class_emb is not None:
            h = torch.cat([h, class_emb], dim=1)

        # Project to latent distribution parameters
        z_mean = self.mean_fc(h)
        z_logvar = self.logvar_fc(h)

        return z_mean, z_logvar

    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def create_encoder(dataset: str, z_dim: int, class_emb_dim: int = 0, **kwargs):
    """Factory function to create encoder based on dataset.

    Args:
        dataset: Dataset name ('mnist' or 'adult')
        z_dim: Latent dimension
        class_emb_dim: Class embedding dimension
        **kwargs: Additional dataset-specific arguments
            - For MNIST: num_ch (default: 32)
            - For Adult: input_dim (default: 108), hidden_dims (default: [256, 128])

    Returns:
        Encoder module

    Raises:
        ValueError: If dataset is not supported
    """
    if dataset.lower() == 'mnist':
        num_ch = kwargs.get('num_ch', 32)
        return MNISTEncoder(z_dim=z_dim, num_ch=num_ch, class_emb_dim=class_emb_dim)

    elif dataset.lower() == 'adult':
        input_dim = kwargs.get('input_dim', 108)
        hidden_dims = kwargs.get('hidden_dims', None)
        return AdultEncoder(
            z_dim=z_dim,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            class_emb_dim=class_emb_dim
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported datasets: 'mnist', 'adult'. "
            "No fallback will be used."
        )
