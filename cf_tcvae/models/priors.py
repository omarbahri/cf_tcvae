"""Class-Conditional Priors for CF-TCVAE.

This module implements Gaussian priors for the latent space with optional
class-conditional parameters. The priors can be updated via EMA (exponential
moving average) during training to track the posterior distribution per class.

Author: CF-TCVAE Team
Date: 2026-02-06
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ClassConditionalGaussianPrior(nn.Module):
    """Class-conditional Gaussian prior P(z|y) with EMA updates.

    For each target class y', maintains a Gaussian prior:
        P(z|y') = N(μ_y', σ²_y')

    The prior parameters can be updated via EMA to track the posterior distribution:
        μ_y' ← (1-m)·μ_y' + m·E[z|y']
        σ²_y' ← (1-m)·σ²_y' + m·Var[z|y']

    where m is the EMA momentum.

    Args:
        latent_dim: Latent space dimensionality
        num_classes: Number of target classes
        init_std: Initial standard deviation for all priors (default: 1.0)
        ema_momentum: EMA momentum for prior updates (default: 0.01)
        min_samples_for_update: Minimum samples per class to perform update (default: 4)
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        init_std: float = 1.0,
        ema_momentum: float = 0.01,
        min_samples_for_update: int = 4
    ):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        if not (0 < ema_momentum < 1):
            raise ValueError(f"ema_momentum must be in (0, 1), got {ema_momentum}")
        if min_samples_for_update < 1:
            raise ValueError(f"min_samples_for_update must be >= 1, got {min_samples_for_update}")

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum
        self.min_samples_for_update = min_samples_for_update

        # Register prior parameters as buffers (saved with model, not optimized)
        # Prior means: [num_classes, latent_dim]
        self.register_buffer(
            'prior_means',
            torch.zeros(num_classes, latent_dim)
        )
        # Prior log-variances: [num_classes, latent_dim]
        self.register_buffer(
            'prior_logvars',
            torch.ones(num_classes, latent_dim) * (2 * torch.log(torch.tensor(init_std)))
        )

    def forward(self, target_classes: torch.Tensor) -> tuple:
        """Get prior parameters for given target classes.

        Args:
            target_classes: Target class labels [batch_size]

        Returns:
            prior_mean: Prior means [batch_size, latent_dim]
            prior_logvar: Prior log-variances [batch_size, latent_dim]
        """
        # Index into prior parameters
        prior_mean = self.prior_means[target_classes]      # [batch_size, latent_dim]
        prior_logvar = self.prior_logvars[target_classes]  # [batch_size, latent_dim]

        return prior_mean, prior_logvar

    @torch.no_grad()
    def update_priors(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        target_classes: torch.Tensor
    ):
        """Update class-conditional priors via EMA.

        For each class present in the batch, computes batch statistics and
        performs EMA update:
            μ_y' ← (1-m)·μ_y' + m·E[z|y']
            σ²_y' ← (1-m)·σ²_y' + m·Var[z|y']

        Args:
            z_mean: Posterior means [batch_size, latent_dim]
            z_logvar: Posterior log-variances [batch_size, latent_dim]
            target_classes: Target class labels [batch_size]
        """
        if not self.training:
            return  # Only update during training

        for class_id in torch.unique(target_classes):
            class_mask = (target_classes == class_id)
            n_class_samples = class_mask.sum().item()

            # Skip if too few samples
            if n_class_samples < self.min_samples_for_update:
                continue

            # Compute batch statistics for this class
            batch_mean = z_mean[class_mask].mean(dim=0)  # [latent_dim]
            batch_var = torch.exp(z_logvar[class_mask]).mean(dim=0)  # [latent_dim]

            # EMA update
            m = self.ema_momentum
            self.prior_means[class_id] = (1 - m) * self.prior_means[class_id] + m * batch_mean
            self.prior_logvars[class_id] = torch.log(
                (1 - m) * torch.exp(self.prior_logvars[class_id]) + m * batch_var
            )

    def kl_divergence(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        target_classes: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence KL(Q(z|x,y')||P(z|y')) for each sample.

        Args:
            z_mean: Posterior means [batch_size, latent_dim]
            z_logvar: Posterior log-variances [batch_size, latent_dim]
            target_classes: Target class labels [batch_size]

        Returns:
            KL divergence per sample [batch_size]
        """
        # Get prior parameters for target classes
        prior_mean, prior_logvar = self.forward(target_classes)

        # KL divergence for diagonal Gaussians:
        # KL(Q||P) = 0.5 * sum_j [exp(logvar_q) / exp(logvar_p) +
        #                         (mu_p - mu_q)^2 / exp(logvar_p) +
        #                         logvar_p - logvar_q - 1]
        kl = 0.5 * torch.sum(
            torch.exp(z_logvar - prior_logvar) +
            (prior_mean - z_mean) ** 2 / torch.exp(prior_logvar) +
            prior_logvar - z_logvar - 1,
            dim=1
        )

        return kl

    def reset_priors(self, init_std: Optional[float] = None):
        """Reset all prior parameters to initial state.

        Args:
            init_std: Initial standard deviation (uses stored value if None)
        """
        if init_std is not None:
            if init_std <= 0:
                raise ValueError(f"init_std must be positive, got {init_std}")
            init_logvar = 2 * torch.log(torch.tensor(init_std))
        else:
            # Use existing init_std from first prior
            init_logvar = self.prior_logvars[0, 0]

        self.prior_means.zero_()
        self.prior_logvars.fill_(init_logvar)

    def get_prior_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about prior parameters.

        Returns:
            Dictionary with prior statistics:
                - prior_means: [num_classes, latent_dim]
                - prior_stds: [num_classes, latent_dim]
                - mean_mean: Scalar (average mean across classes and dims)
                - mean_std: Scalar (average std across classes and dims)
        """
        prior_stds = torch.exp(0.5 * self.prior_logvars)

        return {
            'prior_means': self.prior_means.clone(),
            'prior_stds': prior_stds.clone(),
            'mean_mean': self.prior_means.mean().item(),
            'mean_std': prior_stds.mean().item(),
        }


class StandardGaussianPrior(nn.Module):
    """Standard Gaussian prior P(z) = N(0, I).

    This is a simple baseline prior with no class conditioning.
    Used when use_class_conditional_prior=False.

    Args:
        latent_dim: Latent space dimensionality
    """

    def __init__(self, latent_dim: int):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.latent_dim = latent_dim

        # Register fixed prior parameters as buffers
        self.register_buffer('prior_mean', torch.zeros(latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(latent_dim))

    def forward(self, target_classes: torch.Tensor = None) -> tuple:
        """Get prior parameters (same for all samples).

        Args:
            target_classes: Ignored for standard prior

        Returns:
            prior_mean: Prior means [batch_size, latent_dim]
            prior_logvar: Prior log-variances [batch_size, latent_dim]
        """
        batch_size = 1 if target_classes is None else target_classes.size(0)

        # Expand to batch size
        prior_mean = self.prior_mean.unsqueeze(0).expand(batch_size, -1)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)

        return prior_mean, prior_logvar

    def update_priors(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        target_classes: torch.Tensor
    ):
        """No-op for standard prior (not updated during training)."""
        pass

    def kl_divergence(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        target_classes: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute KL divergence KL(Q(z|x)||N(0,I)) for each sample.

        Args:
            z_mean: Posterior means [batch_size, latent_dim]
            z_logvar: Posterior log-variances [batch_size, latent_dim]
            target_classes: Ignored for standard prior

        Returns:
            KL divergence per sample [batch_size]
        """
        # KL divergence to standard normal:
        # KL(Q||N(0,I)) = 0.5 * sum_j [exp(logvar) + mu^2 - logvar - 1]
        kl = 0.5 * torch.sum(
            torch.exp(z_logvar) + z_mean ** 2 - z_logvar - 1,
            dim=1
        )

        return kl


def create_prior(
    latent_dim: int,
    num_classes: int,
    use_class_conditional: bool = True,
    **kwargs
):
    """Factory function to create prior distribution.

    Args:
        latent_dim: Latent space dimensionality
        num_classes: Number of target classes
        use_class_conditional: Whether to use class-conditional prior
        **kwargs: Additional arguments for class-conditional prior:
            - init_std: Initial standard deviation (default: 1.0)
            - ema_momentum: EMA momentum for updates (default: 0.01)
            - min_samples_for_update: Min samples per class (default: 4)

    Returns:
        Prior module (ClassConditionalGaussianPrior or StandardGaussianPrior)
    """
    if use_class_conditional:
        return ClassConditionalGaussianPrior(
            latent_dim=latent_dim,
            num_classes=num_classes,
            init_std=kwargs.get('init_std', 1.0),
            ema_momentum=kwargs.get('ema_momentum', 0.01),
            min_samples_for_update=kwargs.get('min_samples_for_update', 4)
        )
    else:
        return StandardGaussianPrior(latent_dim=latent_dim)
