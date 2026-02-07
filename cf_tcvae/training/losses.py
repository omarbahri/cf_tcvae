"""Loss functions for CF-TCVAE training.

Implements all loss components from Equation 15 of the paper:
    L = E_Q(z|x,y')[log P(x|z,y) + α·Dist(x,x_cf) + λ·HingeLoss(f(x_cf),y',ε)]
        + (β-1)·TC(z|y') + KL(Q(z|x,y')||P(z|y')) + L_aux(z,y)

All loss functions follow the NO FALLBACKS policy: explicit errors for invalid inputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


def reconstruction_loss_bernoulli(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Bernoulli reconstruction loss for binary data (MNIST).

    Computes: -log P(x|z) = -Σ_i [x_i·log(x̂_i) + (1-x_i)·log(1-x̂_i)]

    Args:
        x_recon: Reconstructed data [batch_size, *dims], range [0, 1]
        x_target: Target data [batch_size, *dims], range [0, 1]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Reconstruction loss (scalar if reduction != 'none')

    Raises:
        ValueError: If reduction is not valid
    """
    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    # Flatten spatial dimensions
    batch_size = x_recon.size(0)
    x_recon_flat = x_recon.view(batch_size, -1)
    x_target_flat = x_target.view(batch_size, -1)

    # Binary cross-entropy per element, then sum over features
    loss_per_sample = F.binary_cross_entropy(
        x_recon_flat, x_target_flat, reduction='none'
    ).sum(dim=1)

    if reduction == 'none':
        return loss_per_sample
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:  # mean
        return loss_per_sample.mean()


def reconstruction_loss_gaussian(
    x_recon_mean: torch.Tensor,
    x_recon_logvar: torch.Tensor,
    x_target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Gaussian reconstruction loss for continuous data (Adult).

    Computes: -log P(x|z) = 0.5·Σ_i [log(2π·σ²_i) + (x_i - μ_i)² / σ²_i]

    Args:
        x_recon_mean: Reconstructed mean [batch_size, *dims]
        x_recon_logvar: Reconstructed log variance [batch_size, *dims]
        x_target: Target data [batch_size, *dims]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Reconstruction loss (scalar if reduction != 'none')

    Raises:
        ValueError: If reduction is not valid or tensor shapes mismatch
    """
    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    if x_recon_mean.shape != x_target.shape or x_recon_logvar.shape != x_target.shape:
        raise ValueError(
            f"Shape mismatch: x_recon_mean {x_recon_mean.shape}, "
            f"x_recon_logvar {x_recon_logvar.shape}, x_target {x_target.shape}"
        )

    # Flatten spatial dimensions
    batch_size = x_recon_mean.size(0)
    mean_flat = x_recon_mean.view(batch_size, -1)
    logvar_flat = x_recon_logvar.view(batch_size, -1)
    target_flat = x_target.view(batch_size, -1)

    # Negative log likelihood: 0.5 * [log(2π·σ²) + (x - μ)² / σ²]
    var = torch.exp(logvar_flat)
    nll = 0.5 * (
        np.log(2 * np.pi) + logvar_flat +
        (target_flat - mean_flat).pow(2) / var
    )

    # Sum over features
    loss_per_sample = nll.sum(dim=1)

    if reduction == 'none':
        return loss_per_sample
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:  # mean
        return loss_per_sample.mean()


def proximity_loss_l1(
    x_cf: torch.Tensor,
    x_orig: torch.Tensor,
    alpha: float,
    reduction: str = 'mean'
) -> torch.Tensor:
    """L1 proximity loss encouraging counterfactuals to stay close to originals.

    Computes: α·||x - x_cf||₁

    Args:
        x_cf: Counterfactual samples [batch_size, *dims]
        x_orig: Original samples [batch_size, *dims]
        alpha: Proximity weight (α)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Proximity loss (scalar if reduction != 'none')

    Raises:
        ValueError: If alpha <= 0, reduction is invalid, or shapes mismatch
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    if x_cf.shape != x_orig.shape:
        raise ValueError(f"Shape mismatch: x_cf {x_cf.shape}, x_orig {x_orig.shape}")

    # Flatten and compute L1 distance
    batch_size = x_cf.size(0)
    cf_flat = x_cf.view(batch_size, -1)
    orig_flat = x_orig.view(batch_size, -1)

    l1_dist = torch.abs(cf_flat - orig_flat).sum(dim=1)
    loss_per_sample = alpha * l1_dist

    if reduction == 'none':
        return loss_per_sample
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:  # mean
        return loss_per_sample.mean()


def hinge_loss_validity(
    logits: torch.Tensor,
    target_class: torch.Tensor,
    epsilon: float,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Hinge loss for validity: ensures counterfactual is classified as target class.

    Computes: max(0, max_{y≠y'} f(x_cf)_y - f(x_cf)_{y'} + ε)

    Args:
        logits: Classifier logits [batch_size, num_classes]
        target_class: Target class labels [batch_size]
        epsilon: Hinge margin (ε)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Hinge loss (scalar if reduction != 'none')

    Raises:
        ValueError: If epsilon <= 0, reduction is invalid, or shapes are wrong
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [batch, num_classes], got shape {logits.shape}")

    batch_size = logits.size(0)
    if target_class.size(0) != batch_size:
        raise ValueError(
            f"Batch size mismatch: logits {batch_size}, target_class {target_class.size(0)}"
        )

    # Handle binary classification (1 logit)
    if logits.size(1) == 1:
        # Convert to 2-class logits: [-logit, logit]
        logits = torch.cat([-logits, logits], dim=1)

    num_classes = logits.size(1)

    # Get logit for target class
    target_logit = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)  # [batch]

    # Get max logit for non-target classes
    mask = F.one_hot(target_class, num_classes).bool()
    max_other_logit = logits.masked_fill(mask, float('-inf')).max(dim=1).values  # [batch]

    # Hinge loss: max(0, max_other - target + ε)
    loss_per_sample = torch.relu(max_other_logit - target_logit + epsilon)

    if reduction == 'none':
        return loss_per_sample
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:  # mean
        return loss_per_sample.mean()


def kl_divergence_diagonal_gaussian(
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussian distributions.

    Computes: KL(Q(z|x,y')||P(z|y')) for diagonal Gaussians

    KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) = 0.5 * Σ_j [
        log(σ²_p,j / σ²_q,j) + (σ²_q,j + (μ_q,j - μ_p,j)²) / σ²_p,j - 1
    ]

    Args:
        z_mean: Posterior mean [batch_size, z_dim]
        z_logvar: Posterior log variance [batch_size, z_dim]
        prior_mean: Prior mean [batch_size, z_dim] or [z_dim]
        prior_logvar: Prior log variance [batch_size, z_dim] or [z_dim]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        KL divergence (scalar if reduction != 'none')

    Raises:
        ValueError: If reduction is invalid or shapes are incompatible
    """
    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    if z_mean.shape != z_logvar.shape:
        raise ValueError(f"Shape mismatch: z_mean {z_mean.shape}, z_logvar {z_logvar.shape}")

    # Broadcast prior if needed (e.g., [z_dim] -> [batch_size, z_dim])
    if prior_mean.dim() == 1:
        prior_mean = prior_mean.unsqueeze(0).expand_as(z_mean)
    if prior_logvar.dim() == 1:
        prior_logvar = prior_logvar.unsqueeze(0).expand_as(z_logvar)

    if z_mean.shape != prior_mean.shape or z_mean.shape != prior_logvar.shape:
        raise ValueError(
            f"Shape mismatch after broadcasting: z_mean {z_mean.shape}, "
            f"prior_mean {prior_mean.shape}, prior_logvar {prior_logvar.shape}"
        )

    # KL divergence formula for diagonal Gaussians
    var_q = torch.exp(z_logvar)
    var_p = torch.exp(prior_logvar)

    kl_per_dim = 0.5 * (
        prior_logvar - z_logvar +
        (var_q + (z_mean - prior_mean).pow(2)) / var_p - 1.0
    )

    # Sum over latent dimensions
    kl_per_sample = kl_per_dim.sum(dim=1)

    if reduction == 'none':
        return kl_per_sample
    elif reduction == 'sum':
        return kl_per_sample.sum()
    else:  # mean
        return kl_per_sample.mean()


def auxiliary_classifier_loss(
    aux_logits: torch.Tensor,
    original_class: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Auxiliary classifier loss to prevent mode collapse.

    Ensures latent code retains information about original class y.

    Computes: L_aux = CrossEntropy(aux_classifier(z), y)

    Args:
        aux_logits: Auxiliary classifier logits [batch_size, num_classes]
        original_class: Original class labels [batch_size]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Auxiliary loss (scalar if reduction != 'none')

    Raises:
        ValueError: If reduction is invalid or shapes are wrong
    """
    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    if aux_logits.dim() != 2:
        raise ValueError(
            f"aux_logits must be 2D [batch, num_classes], got shape {aux_logits.shape}"
        )

    batch_size = aux_logits.size(0)
    if original_class.size(0) != batch_size:
        raise ValueError(
            f"Batch size mismatch: aux_logits {batch_size}, original_class {original_class.size(0)}"
        )

    return F.cross_entropy(aux_logits, original_class, reduction=reduction)


def compute_validity_rate(
    logits: torch.Tensor,
    target_class: torch.Tensor
) -> torch.Tensor:
    """Compute validity rate: fraction of counterfactuals classified as target class.

    Validity = (1/N) Σ 1[argmax f(x_cf) = y']

    Args:
        logits: Classifier logits [batch_size, num_classes]
        target_class: Target class labels [batch_size]

    Returns:
        Validity rate [0, 1]

    Raises:
        ValueError: If shapes are wrong
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [batch, num_classes], got shape {logits.shape}")

    batch_size = logits.size(0)
    if target_class.size(0) != batch_size:
        raise ValueError(
            f"Batch size mismatch: logits {batch_size}, target_class {target_class.size(0)}"
        )

    # Handle binary classification (1 logit)
    if logits.size(1) == 1:
        logits = torch.cat([-logits, logits], dim=1)

    predicted = logits.argmax(dim=1)
    valid = (predicted == target_class).float()

    return valid.mean()


class LossAggregator:
    """Aggregates all CF-TCVAE loss components with proper weighting.

    Implements Equation 15:
        L = recon + α·proximity + λ·validity + KL + (β-1)·TC + aux_weight·aux

    Attributes:
        beta: TC regularization weight (β)
        alpha_proximity: Proximity loss weight (α)
        lambda_validity: Validity loss weight (λ)
        epsilon: Hinge loss margin (ε)
        aux_weight: Auxiliary classifier loss weight
        aux_warmup_epochs: Number of epochs before auxiliary loss is activated
        likelihood: 'bernoulli' or 'gaussian'
    """

    def __init__(
        self,
        beta: float,
        alpha_proximity: float,
        lambda_validity: float,
        epsilon: float,
        aux_weight: float,
        aux_warmup_epochs: int,
        likelihood: str = 'bernoulli'
    ):
        """Initialize loss aggregator.

        Args:
            beta: TC regularization weight (must be > 0)
            alpha_proximity: Proximity loss weight (must be >= 0)
            lambda_validity: Validity loss weight (must be >= 0)
            epsilon: Hinge loss margin (must be > 0)
            aux_weight: Auxiliary classifier loss weight (must be >= 0)
            aux_warmup_epochs: Epochs before auxiliary loss activates (must be >= 0)
            likelihood: 'bernoulli' (MNIST) or 'gaussian' (Adult)

        Raises:
            ValueError: If any parameter is invalid
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if alpha_proximity < 0:
            raise ValueError(f"alpha_proximity must be non-negative, got {alpha_proximity}")
        if lambda_validity < 0:
            raise ValueError(f"lambda_validity must be non-negative, got {lambda_validity}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if aux_weight < 0:
            raise ValueError(f"aux_weight must be non-negative, got {aux_weight}")
        if aux_warmup_epochs < 0:
            raise ValueError(f"aux_warmup_epochs must be non-negative, got {aux_warmup_epochs}")
        if likelihood not in ['bernoulli', 'gaussian']:
            raise ValueError(f"likelihood must be 'bernoulli' or 'gaussian', got '{likelihood}'")

        self.beta = beta
        self.alpha_proximity = alpha_proximity
        self.lambda_validity = lambda_validity
        self.epsilon = epsilon
        self.aux_weight = aux_weight
        self.aux_warmup_epochs = aux_warmup_epochs
        self.likelihood = likelihood

    def aggregate(
        self,
        recon_loss: torch.Tensor,
        proximity_loss: torch.Tensor,
        validity_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        tc_loss: torch.Tensor,
        aux_loss: Optional[torch.Tensor],
        current_epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Aggregate all loss components into total loss.

        Args:
            recon_loss: Reconstruction loss (already scaled)
            proximity_loss: Proximity loss (already scaled by α)
            validity_loss: Validity loss (already scaled by λ)
            kl_loss: KL divergence loss
            tc_loss: Total correlation loss (will be scaled by β-1)
            aux_loss: Auxiliary classifier loss (or None if disabled)
            current_epoch: Current training epoch (for warmup)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual components
        """
        # Scale TC loss by (β - 1)
        scaled_tc_loss = (self.beta - 1.0) * tc_loss

        # Apply auxiliary warmup
        scaled_aux_loss = torch.tensor(0.0, device=recon_loss.device)
        if aux_loss is not None and current_epoch >= self.aux_warmup_epochs:
            scaled_aux_loss = self.aux_weight * aux_loss

        # Total loss
        total_loss = (
            recon_loss +
            proximity_loss +
            validity_loss +
            kl_loss +
            scaled_tc_loss +
            scaled_aux_loss
        )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'proximity': proximity_loss.item(),
            'validity': validity_loss.item(),
            'kl': kl_loss.item(),
            'tc': tc_loss.item(),
            'tc_scaled': scaled_tc_loss.item(),
            'aux': aux_loss.item() if aux_loss is not None else 0.0,
            'aux_scaled': scaled_aux_loss.item(),
        }

        return total_loss, loss_dict
