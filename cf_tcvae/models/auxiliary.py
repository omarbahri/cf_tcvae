"""Auxiliary Classifier for CF-TCVAE.

This module implements the auxiliary classifier head that predicts the original
class label from the latent code. This encourages the latent space to retain
information about the original class, preventing mode collapse.

The auxiliary classifier is trained with warmup: it only starts contributing to
the loss after a specified number of epochs to allow the VAE to stabilize first.

Author: CF-TCVAE Team
Date: 2026-02-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryClassifier(nn.Module):
    """Auxiliary classifier head for predicting original class from latent code.

    Architecture:
        z → [hidden] → logits

    The classifier takes latent codes as input and predicts the original class label.
    This helps preserve class information in the latent space and prevents the
    encoder from collapsing to a single mode.

    Args:
        latent_dim: Latent space dimensionality
        num_classes: Number of classes to predict
        hidden_dim: Hidden layer dimension (0 for linear classifier)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if hidden_dim < 0:
            raise ValueError(f"hidden_dim must be non-negative, got {hidden_dim}")
        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Build classifier network
        if hidden_dim > 0:
            # Two-layer classifier with hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Linear(latent_dim, num_classes)

        self._init_weights()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict class logits from latent code.

        Args:
            z: Latent codes [batch_size, latent_dim]

        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        if z.size(1) != self.latent_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.latent_dim}, got {z.size(1)}"
            )

        logits = self.classifier(z)
        return logits

    def compute_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute cross-entropy loss for auxiliary classification.

        Args:
            z: Latent codes [batch_size, latent_dim]
            labels: True class labels [batch_size]
            reduction: Loss reduction ('mean', 'sum', or 'none')

        Returns:
            Cross-entropy loss
        """
        logits = self.forward(z)
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        return loss

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Predict class labels from latent code.

        Args:
            z: Latent codes [batch_size, latent_dim]

        Returns:
            predictions: Predicted class labels [batch_size]
        """
        with torch.no_grad():
            logits = self.forward(z)
            predictions = logits.argmax(dim=1)
        return predictions

    def accuracy(self, z: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy.

        Args:
            z: Latent codes [batch_size, latent_dim]
            labels: True class labels [batch_size]

        Returns:
            Accuracy (fraction of correct predictions)
        """
        predictions = self.predict(z)
        correct = (predictions == labels).float().sum()
        accuracy = correct / labels.size(0)
        return accuracy.item()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)


class AuxiliaryClassifierWithWarmup:
    """Wrapper for auxiliary classifier with warmup schedule.

    The auxiliary loss only starts contributing after a warmup period.
    This allows the VAE to stabilize before adding the auxiliary objective.

    Args:
        classifier: AuxiliaryClassifier module
        warmup_epochs: Number of epochs before auxiliary loss is activated
        weight: Weight multiplier for auxiliary loss (default: 1.0)
    """

    def __init__(
        self,
        classifier: AuxiliaryClassifier,
        warmup_epochs: int = 10,
        weight: float = 1.0
    ):
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")

        self.classifier = classifier
        self.warmup_epochs = warmup_epochs
        self.weight = weight
        self.current_epoch = 0

    def step_epoch(self):
        """Increment the current epoch counter."""
        self.current_epoch += 1

    def is_active(self) -> bool:
        """Check if auxiliary loss is active (past warmup period)."""
        return self.current_epoch >= self.warmup_epochs

    def compute_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute weighted auxiliary loss (returns 0 during warmup).

        Args:
            z: Latent codes [batch_size, latent_dim]
            labels: True class labels [batch_size]
            reduction: Loss reduction ('mean', 'sum', or 'none')

        Returns:
            Weighted auxiliary loss (0 if not active)
        """
        if not self.is_active():
            return torch.tensor(0.0, device=z.device, requires_grad=False)

        loss = self.classifier.compute_loss(z, labels, reduction=reduction)
        return self.weight * loss

    def __call__(self, *args, **kwargs):
        """Forward pass through classifier."""
        return self.classifier(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Move classifier to device."""
        self.classifier = self.classifier.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        self.classifier.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.classifier.eval()
        return self


def create_auxiliary_classifier(
    latent_dim: int,
    num_classes: int,
    use_warmup: bool = True,
    warmup_epochs: int = 10,
    weight: float = 1.0,
    hidden_dim: int = 0,
    dropout: float = 0.0
):
    """Factory function to create auxiliary classifier with optional warmup.

    Args:
        latent_dim: Latent space dimensionality
        num_classes: Number of classes to predict
        use_warmup: Whether to use warmup schedule
        warmup_epochs: Number of warmup epochs (if use_warmup=True)
        weight: Weight multiplier for auxiliary loss
        hidden_dim: Hidden layer dimension (0 for linear classifier)
        dropout: Dropout probability

    Returns:
        AuxiliaryClassifier or AuxiliaryClassifierWithWarmup
    """
    classifier = AuxiliaryClassifier(
        latent_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    if use_warmup:
        return AuxiliaryClassifierWithWarmup(
            classifier=classifier,
            warmup_epochs=warmup_epochs,
            weight=weight
        )
    else:
        return classifier
