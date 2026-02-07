"""Class-Conditional Total Correlation Estimator.

This module implements class-conditional TC estimation using the minibatch-weighted
estimator from Chen et al. 2018 β-TCVAE. The estimator groups samples by target class
and computes TC within each class, then averages across classes.

Author: CF-TCVAE Team
Date: 2026-02-06
"""

import math
import torch
import torch.nn as nn
from typing import Dict


class MinibatchTCEstimator(nn.Module):
    """Class-conditional Total Correlation estimator with memory bank support.

    Implements Chen et al. 2018 minibatch TC estimation with class-conditional
    grouping. For each class present in the batch, computes TC using only samples
    from that class, then averages TC across classes.

    Memory banks augment small per-class sample counts to ensure robust TC estimation
    even when batch size is unevenly distributed across classes.

    Args:
        latent_dim: Latent space dimensionality
        num_classes: Number of target classes
        use_class_conditional: Whether to use class-conditional TC estimation
        min_samples_per_class: Minimum samples per class for TC estimation
        bank_capacity_per_class: Memory bank capacity per class
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        use_class_conditional: bool = True,
        min_samples_per_class: int = 8,
        bank_capacity_per_class: int = 256
    ):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if min_samples_per_class < 2:
            raise ValueError(f"min_samples_per_class must be >= 2, got {min_samples_per_class}")
        if bank_capacity_per_class < min_samples_per_class:
            raise ValueError(
                f"bank_capacity_per_class ({bank_capacity_per_class}) must be >= "
                f"min_samples_per_class ({min_samples_per_class})"
            )

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_class_conditional = use_class_conditional
        self.min_samples_per_class = min_samples_per_class
        self.bank_capacity_per_class = bank_capacity_per_class

        # Memory banks for each class to store (z_mean, z_logvar) pairs
        # Use buffers so they're saved with the model but don't require gradients
        if use_class_conditional:
            # Memory bank for z_mean: [num_classes, bank_capacity, latent_dim]
            self.register_buffer(
                'memory_z_mean',
                torch.zeros(num_classes, bank_capacity_per_class, latent_dim)
            )
            # Memory bank for z_logvar: [num_classes, bank_capacity, latent_dim]
            self.register_buffer(
                'memory_z_logvar',
                torch.zeros(num_classes, bank_capacity_per_class, latent_dim)
            )
            # Bank pointers for circular buffer: [num_classes]
            self.register_buffer('bank_ptr', torch.zeros(num_classes, dtype=torch.long))
            # Bank fill status: [num_classes] - tracks how many samples are stored
            self.register_buffer('bank_size', torch.zeros(num_classes, dtype=torch.long))

    def forward(
        self,
        z_samples: torch.Tensor,
        target_classes: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute class-conditional total correlation loss.

        Args:
            z_samples: Sampled latent codes [batch_size, latent_dim]
            target_classes: Target class labels [batch_size]
            z_mean: Posterior means [batch_size, latent_dim]
            z_logvar: Posterior log-variances [batch_size, latent_dim]

        Returns:
            Total correlation loss tensor (scalar)
        """
        if not self.use_class_conditional:
            # Standard TC estimation without class conditioning
            return self._compute_standard_tc(z_samples, z_mean, z_logvar)

        # Update memory banks with current batch
        if self.training:
            self._update_memory_banks(z_mean, z_logvar, target_classes)

        # Compute class-conditional TC
        return self._compute_class_conditional_tc(z_samples, z_mean, z_logvar, target_classes)

    def _compute_standard_tc(
        self,
        z_samples: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """Standard TC estimation without class conditioning (Chen et al. 2018)."""
        MB, z_dim = z_samples.shape

        # Reshape for pairwise comparisons
        z_sample = z_samples.view(MB, 1, z_dim)         # (MB, 1, z_dim)
        z_q_mean = z_mean.view(1, MB, z_dim)           # (1, MB, z_dim)
        z_q_logvar = z_logvar.view(1, MB, z_dim)       # (1, MB, z_dim)

        # Numerical stability for variance
        denom = torch.exp(z_q_logvar).clamp(min=1e-6)

        # log q(z^{(i)} | x^{(n)}) for all i,n,j
        log_qz_i = -0.5 * (
            math.log(2.0 * math.pi) +
            z_q_logvar +
            (z_sample - z_q_mean) ** 2 / denom
        )  # (MB, MB, z_dim)

        # Device-safe log(MB)
        log_mb = torch.log(torch.tensor(MB, dtype=z_samples.dtype, device=z_samples.device))

        # log q(z_j) evaluated at each sampled z^{(i)}:
        log_qz_j = torch.logsumexp(log_qz_i, dim=1) - log_mb     # (MB, z_dim)

        # log q(z) evaluated at each sampled z^{(i)}:
        log_qz = torch.logsumexp(log_qz_i.sum(2), dim=1) - log_mb  # (MB,)

        # TC = log q(z) - sum_j log q(z_j)
        tc_loss = (log_qz - log_qz_j.sum(dim=1))  # (MB,)

        return tc_loss.mean()

    def _compute_class_conditional_tc(
        self,
        z_samples: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        target_classes: torch.Tensor
    ) -> torch.Tensor:
        """Class-conditional TC estimation with memory bank augmentation."""
        device = z_samples.device

        # Equal per-class weighting: accumulate per-class TC means, then average
        class_tc_means = []

        # Process each class separately
        for class_id in torch.unique(target_classes):
            class_mask = (target_classes == class_id)
            n_class_samples = int(class_mask.sum().item())
            if n_class_samples == 0:
                continue

            # Get samples for this class
            z_class = z_samples[class_mask]           # [n_class, latent_dim]
            z_mean_class = z_mean[class_mask]         # [n_class, latent_dim]
            z_logvar_class = z_logvar[class_mask]     # [n_class, latent_dim]

            # Augment with memory bank if needed to reach minimum per-class count
            if n_class_samples < self.min_samples_per_class:
                z_mean_aug, z_logvar_aug = self._get_augmented_class_samples(
                    class_id, z_mean_class, z_logvar_class
                )
            else:
                z_mean_aug, z_logvar_aug = z_mean_class, z_logvar_class

            # Compute TC for this class using Chen et al. 2018 method
            class_tc_vector = self._compute_tc_for_class(z_class, z_mean_aug, z_logvar_aug)

            # Per-class scalar by averaging within the class
            class_tc_means.append(class_tc_vector.mean())

        if not class_tc_means:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Equal mean over present classes
        return torch.stack(class_tc_means).mean()

    def _get_augmented_class_samples(
        self,
        class_id: int,
        z_mean_class: torch.Tensor,
        z_logvar_class: torch.Tensor
    ) -> tuple:
        """Augment class samples by randomly drawing from the memory bank.

        If current class count is below min_samples_per_class, randomly sample
        additional (mu, logvar) entries from the per-class bank to reach the minimum.
        """
        device = z_mean_class.device
        n_current = int(z_mean_class.size(0))
        bank_size = int(self.bank_size[class_id].item())

        # Nothing to add if enough current samples or bank empty
        if n_current >= self.min_samples_per_class or bank_size == 0:
            return z_mean_class, z_logvar_class

        need = max(0, self.min_samples_per_class - n_current)
        k = min(need, bank_size)

        stored_mean = self.memory_z_mean[class_id, :bank_size]
        stored_logvar = self.memory_z_logvar[class_id, :bank_size]

        # Randomly sample k entries from the bank
        idx = torch.randperm(bank_size, device=device)[:k]
        extra_mean = stored_mean.index_select(0, idx)
        extra_logvar = stored_logvar.index_select(0, idx)

        aug_mean = torch.cat([z_mean_class, extra_mean], dim=0)
        aug_logvar = torch.cat([z_logvar_class, extra_logvar], dim=0)

        return aug_mean, aug_logvar

    def _compute_tc_for_class(
        self,
        z_samples: torch.Tensor,
        z_mean_aug: torch.Tensor,
        z_logvar_aug: torch.Tensor
    ) -> torch.Tensor:
        """Compute TC for a single class using Chen et al. 2018 method."""
        n_samples = z_samples.size(0)
        n_aug = z_mean_aug.size(0)

        if n_aug < 2:
            # Not enough samples for TC estimation, return zero
            return torch.zeros(n_samples, device=z_samples.device)

        z_dim = z_samples.size(1)

        # Reshape for pairwise comparisons with augmented data
        z_sample = z_samples.view(n_samples, 1, z_dim)        # (n_samples, 1, z_dim)
        z_q_mean = z_mean_aug.view(1, n_aug, z_dim)           # (1, n_aug, z_dim)
        z_q_logvar = z_logvar_aug.view(1, n_aug, z_dim)       # (1, n_aug, z_dim)

        # Numerical stability for variance
        denom = torch.exp(z_q_logvar).clamp(min=1e-6)

        # log q(z^{(i)} | x^{(n)}) for all i,n,j
        log_qz_i = -0.5 * (
            math.log(2.0 * math.pi) +
            z_q_logvar +
            (z_sample - z_q_mean) ** 2 / denom
        )  # (n_samples, n_aug, z_dim)

        # Device-safe log(n_aug)
        log_n_aug = torch.log(torch.tensor(n_aug, dtype=z_samples.dtype, device=z_samples.device))

        # log q(z_j) evaluated at each sampled z^{(i)}:
        log_qz_j = torch.logsumexp(log_qz_i, dim=1) - log_n_aug     # (n_samples, z_dim)

        # log q(z) evaluated at each sampled z^{(i)}:
        log_qz = torch.logsumexp(log_qz_i.sum(2), dim=1) - log_n_aug  # (n_samples,)

        # TC = log q(z) - sum_j log q(z_j)
        tc_loss = (log_qz - log_qz_j.sum(dim=1))  # (n_samples,)

        return tc_loss

    @torch.no_grad()
    def _update_memory_banks(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        target_classes: torch.Tensor
    ):
        """Update memory banks with current batch samples."""
        for class_id in torch.unique(target_classes):
            class_mask = (target_classes == class_id)
            if not class_mask.any():
                continue

            # Detach to avoid tracking gradients in buffers
            class_mean = z_mean[class_mask].detach()    # [n_class, latent_dim]
            class_logvar = z_logvar[class_mask].detach() # [n_class, latent_dim]
            n_class = class_mean.size(0)

            # Update memory bank for this class (circular buffer)
            ptr = self.bank_ptr[class_id].item()
            bank_size = self.bank_size[class_id].item()

            for i in range(n_class):
                # Store sample in memory bank
                self.memory_z_mean[class_id, ptr] = class_mean[i]
                self.memory_z_logvar[class_id, ptr] = class_logvar[i]

                # Update pointer and size
                ptr = (ptr + 1) % self.bank_capacity_per_class
                bank_size = min(bank_size + 1, self.bank_capacity_per_class)

            # Update buffers
            self.bank_ptr[class_id] = ptr
            self.bank_size[class_id] = bank_size

    def reset_memory_banks(self):
        """Reset all memory banks (useful for new training runs)."""
        if self.use_class_conditional:
            self.bank_ptr.zero_()
            self.bank_size.zero_()
            self.memory_z_mean.zero_()
            self.memory_z_logvar.zero_()

    def get_memory_bank_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about memory bank usage."""
        if not self.use_class_conditional:
            return {}

        return {
            'bank_sizes': self.bank_size.clone(),
            'bank_utilization': self.bank_size.float() / self.bank_capacity_per_class,
            'total_stored_samples': self.bank_size.sum(),
            'avg_samples_per_class': self.bank_size.float().mean()
        }
