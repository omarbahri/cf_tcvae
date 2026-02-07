"""Decoder architectures for CF-TCVAE.

This module implements clean decoder architectures for MNIST (CNN) and Adult (MLP)
without any FiLM layer dependencies. All class conditioning is done via concatenation
before the first layer.

Author: CF-TCVAE Team
Date: 2026-02-06
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    """CNN decoder for MNIST-style image data.

    Architecture:
    - Linear projection from z to spatial representation (2x2)
    - 5 transposed convolutional blocks with upsampling (2->5->7->14->28)
    - Class embedding concatenated with latent code before projection
    - Outputs mu and logvar for reconstruction distribution

    Args:
        z_dim: Latent dimension
        in_channels: Number of base channels (default: 32)
        data_channels: Number of image channels (1 for MNIST, 3 for color)
        class_emb_dim: Dimension of class embeddings (0 for no conditioning)
    """

    def __init__(self, z_dim: int, in_channels: int = 32, data_channels: int = 1,
                 class_emb_dim: int = 0):
        super().__init__()

        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if data_channels <= 0:
            raise ValueError(f"data_channels must be positive, got {data_channels}")
        if class_emb_dim < 0:
            raise ValueError(f"class_emb_dim must be non-negative, got {class_emb_dim}")

        self.z_dim = z_dim
        self.in_channels = in_channels
        self.data_channels = data_channels
        self.class_emb_dim = class_emb_dim

        # Channel progression (reverse of encoder): [128, 96, 64, 32]
        ch = [z_dim // 4, in_channels * 4, in_channels * 3, in_channels * 2, in_channels]
        self.z_shape = 2  # Initial spatial size

        # Project latent code (+ optional class embedding) to spatial representation
        if class_emb_dim > 0:
            self.z_projection = nn.Sequential(
                nn.Linear(z_dim + class_emb_dim, ch[0] * self.z_shape * self.z_shape),
                nn.ReLU()
            )
        else:
            self.z_projection = nn.Sequential(
                nn.Linear(z_dim, ch[0] * self.z_shape * self.z_shape),
                nn.ReLU()
            )

        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(ch[0], ch[1], 3, stride=2, padding=0)  # 2->5
        self.deconv2 = nn.ConvTranspose2d(ch[1], ch[2], 3, stride=1, padding=0)  # 5->7
        self.deconv3 = nn.ConvTranspose2d(ch[2], ch[3], 3, stride=1, padding=1)  # 7->7
        self.deconv4 = nn.ConvTranspose2d(ch[3], ch[4], 4, stride=2, padding=1)  # 7->14
        self.deconv5 = nn.ConvTranspose2d(ch[4], data_channels, 4, stride=2, padding=1)  # 14->28

        # Output activation (Sigmoid for MNIST, Identity for others)
        if data_channels == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

        self._init_weights()

    def forward(self, z: torch.Tensor, class_emb: torch.Tensor = None) -> tuple:
        """Forward pass through decoder.

        Args:
            z: Latent code [batch_size, z_dim]
            class_emb: Class embeddings [batch_size, class_emb_dim] (optional)

        Returns:
            x_mean: Reconstruction mean [batch_size, channels, height, width]
            x_logvar: Reconstruction log-variance [batch_size, channels, height, width]
        """
        # Handle class conditioning
        if self.class_emb_dim > 0:
            if class_emb is None:
                raise ValueError("class_emb must be provided when class_emb_dim > 0")
            if class_emb.size(1) != self.class_emb_dim:
                raise ValueError(
                    f"class_emb has dimension {class_emb.size(1)}, expected {self.class_emb_dim}"
                )
            # Concatenate z with class embeddings
            z_input = torch.cat([z, class_emb], dim=1)
        else:
            z_input = z

        # Project to spatial representation
        h = self.z_projection(z_input)
        h = h.view(-1, self.z_dim // 4, self.z_shape, self.z_shape)  # [B, ch[0], 2, 2]

        # Transposed conv layers with ReLU (except final layer)
        h = F.relu(self.deconv1(h))  # [B, 128, 5, 5]
        h = F.relu(self.deconv2(h))  # [B, 96, 7, 7]
        h = F.relu(self.deconv3(h))  # [B, 64, 7, 7]
        h = F.relu(self.deconv4(h))  # [B, 32, 14, 14]
        h = self.deconv5(h)          # [B, 1, 28, 28]

        # Apply output activation
        x_mean = self.output_activation(h)

        # For Bernoulli likelihood, logvar is constant (log(2π))
        x_logvar = -torch.ones_like(x_mean) * math.log(2 * math.pi)

        return x_mean, x_logvar

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)


class MLPDecoder(nn.Module):
    """MLP decoder for tabular data (e.g., Adult dataset).

    Architecture:
    - Linear projection from z (+ optional class embedding) to hidden layers
    - 2 hidden layers with ReLU activations
    - Outputs mu and logvar for reconstruction distribution (Gaussian likelihood)

    Args:
        z_dim: Latent dimension
        output_dim: Output feature dimension
        hidden_dims: List of hidden layer dimensions (default: [128, 256])
        class_emb_dim: Dimension of class embeddings (0 for no conditioning)
    """

    def __init__(self, z_dim: int, output_dim: int, hidden_dims: list = None,
                 class_emb_dim: int = 0):
        super().__init__()

        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if class_emb_dim < 0:
            raise ValueError(f"class_emb_dim must be non-negative, got {class_emb_dim}")

        self.z_dim = z_dim
        self.output_dim = output_dim
        self.class_emb_dim = class_emb_dim

        # Default hidden dimensions (reverse of encoder)
        if hidden_dims is None:
            hidden_dims = [128, 256]

        if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty list")

        self.hidden_dims = hidden_dims

        # Input dimension includes z and optional class embedding
        input_dim = z_dim + (class_emb_dim if class_emb_dim > 0 else 0)

        # Build decoder layers
        dims = [input_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = nn.ReLU()

        # Output heads for mean and logvar (Gaussian likelihood)
        final_dim = hidden_dims[-1]
        self.x_mean = nn.Linear(final_dim, output_dim)
        self.x_logvar = nn.Linear(final_dim, output_dim)

        self._init_weights()

    def forward(self, z: torch.Tensor, class_emb: torch.Tensor = None) -> tuple:
        """Forward pass through decoder.

        Args:
            z: Latent code [batch_size, z_dim]
            class_emb: Class embeddings [batch_size, class_emb_dim] (optional)

        Returns:
            x_mean: Reconstruction mean [batch_size, output_dim]
            x_logvar: Reconstruction log-variance [batch_size, output_dim]
        """
        # Handle class conditioning
        if self.class_emb_dim > 0:
            if class_emb is None:
                raise ValueError("class_emb must be provided when class_emb_dim > 0")
            if class_emb.size(1) != self.class_emb_dim:
                raise ValueError(
                    f"class_emb has dimension {class_emb.size(1)}, expected {self.class_emb_dim}"
                )
            # Concatenate z with class embeddings
            z_input = torch.cat([z, class_emb], dim=1)
        else:
            z_input = z

        # Process through hidden layers
        out = z_input
        for layer in self.hidden_layers:
            out = self.activation(layer(out))

        # Generate output mean and logvar
        x_mean = self.x_mean(out)
        x_logvar = self.x_logvar(out)

        # Clamp logvar to prevent numerical instability
        # Prevents model from "cheating" by claiming infinite confidence (variance → 0)
        x_logvar = torch.clamp(x_logvar, min=-6.0, max=4.0)

        return x_mean, x_logvar

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)


def create_decoder(dataset: str, z_dim: int, **kwargs):
    """Factory function to create decoder for specified dataset.

    Args:
        dataset: Dataset name ('mnist' or 'adult')
        z_dim: Latent dimension
        **kwargs: Additional architecture parameters
            For MNIST:
                - in_channels: Base conv filters (default: 32)
                - data_channels: Image channels (default: 1)
                - class_emb_dim: Class embedding dimension (default: 0)
            For Adult:
                - output_dim: Output feature dimension (required)
                - hidden_dims: Hidden layer dimensions (default: [128, 256])
                - class_emb_dim: Class embedding dimension (default: 0)

    Returns:
        Decoder instance

    Raises:
        ValueError: If dataset is unknown or required kwargs are missing
    """
    dataset = dataset.lower()

    if dataset == 'mnist':
        return CNNDecoder(
            z_dim=z_dim,
            in_channels=kwargs.get('in_channels', 32),
            data_channels=kwargs.get('data_channels', 1),
            class_emb_dim=kwargs.get('class_emb_dim', 0)
        )
    elif dataset == 'adult':
        if 'output_dim' not in kwargs:
            raise ValueError("output_dim is required for Adult decoder")
        return MLPDecoder(
            z_dim=z_dim,
            output_dim=kwargs['output_dim'],
            hidden_dims=kwargs.get('hidden_dims', [128, 256]),
            class_emb_dim=kwargs.get('class_emb_dim', 0)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'mnist' or 'adult'")
