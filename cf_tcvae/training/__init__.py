"""Training utilities for CF-TCVAE.

This module provides:
- Loss functions (reconstruction, proximity, validity, KL, TC, auxiliary)
- Unified training loop (single-phase, no progressive training)
- Checkpoint management (save/load with best model tracking)
- Learning rate scheduling and early stopping
"""

from cf_tcvae.training.losses import (
    reconstruction_loss_bernoulli,
    reconstruction_loss_gaussian,
    proximity_loss_l1,
    hinge_loss_validity,
    kl_divergence_diagonal_gaussian,
    auxiliary_classifier_loss,
    compute_validity_rate,
    LossAggregator,
)

from cf_tcvae.training.checkpoint import (
    CheckpointManager,
    save_model_only,
    load_model_only,
)

from cf_tcvae.training.trainer import CFTCVAETrainer


__all__ = [
    # Loss functions
    'reconstruction_loss_bernoulli',
    'reconstruction_loss_gaussian',
    'proximity_loss_l1',
    'hinge_loss_validity',
    'kl_divergence_diagonal_gaussian',
    'auxiliary_classifier_loss',
    'compute_validity_rate',
    'LossAggregator',
    # Checkpoint utilities
    'CheckpointManager',
    'save_model_only',
    'load_model_only',
    # Trainer
    'CFTCVAETrainer',
]
