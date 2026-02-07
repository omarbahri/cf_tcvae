"""Unified single-phase training loop for CF-TCVAE.

NO phase1/phase2 logic - trains all components jointly from the start.
Implements Equation 15 from the paper with proper EMA prior updates.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cf_tcvae.models import CFTCVAE
from cf_tcvae.training.losses import (
    reconstruction_loss_bernoulli,
    reconstruction_loss_gaussian,
    proximity_loss_l1,
    hinge_loss_validity,
    kl_divergence_diagonal_gaussian,
    auxiliary_classifier_loss,
    compute_validity_rate,
    LossAggregator
)
from cf_tcvae.training.checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


class CFTCVAETrainer:
    """Unified trainer for CF-TCVAE (single-phase training).

    Features:
    - Unified loss computation (Equation 15)
    - EMA class-conditional prior updates
    - Learning rate scheduling with ReduceLROnPlateau
    - Checkpoint saving (best model based on validation loss)
    - Progress tracking with tqdm
    - Comprehensive logging of all loss components

    NO phase1/phase2 logic - all components train jointly.
    """

    def __init__(
        self,
        model: CFTCVAE,
        classifier: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        checkpoint_dir: str,
        device: str = 'cuda',
        log_interval: int = 50
    ):
        """Initialize trainer.

        Args:
            model: CF-TCVAE model
            classifier: Pre-trained frozen black-box classifier
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object (MNISTConfig or AdultConfig)
            checkpoint_dir: Directory to save checkpoints
            device: 'cuda' or 'cpu'
            log_interval: Log metrics every N batches

        Raises:
            ValueError: If config is invalid or missing required parameters
        """
        self.model = model.to(device)
        self.classifier = classifier.to(device)
        self.classifier.eval()  # Always in eval mode (frozen)

        # Freeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_interval = log_interval

        # Validate config
        self._validate_config(config)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0
        )

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=True,
            min_lr=1e-6
        )

        # Setup loss aggregator
        self.loss_aggregator = LossAggregator(
            beta=config.beta,
            alpha_proximity=config.alpha_proximity,
            lambda_validity=config.lambda_validity,
            epsilon=config.epsilon,
            aux_weight=config.aux_weight,
            aux_warmup_epochs=config.aux_warmup_epochs,
            likelihood=config.likelihood
        )

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_best_only=True,
            best_metric_name='val_loss',
            best_metric_mode='min'
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(f"CFTCVAETrainer initialized: device={device}, "
                    f"lr={config.lr}, beta={config.beta}")

    def _validate_config(self, config):
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required = [
            'lr', 'beta', 'alpha_proximity', 'lambda_validity', 'epsilon',
            'aux_weight', 'aux_warmup_epochs', 'likelihood', 'lr_patience',
            'lr_factor', 'max_epochs'
        ]

        for param in required:
            if not hasattr(config, param):
                raise ValueError(f"Config is missing required parameter: {param}")

        # Validate parameter values
        if config.lr <= 0:
            raise ValueError(f"lr must be positive, got {config.lr}")
        if config.beta <= 0:
            raise ValueError(f"beta must be positive, got {config.beta}")
        if config.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {config.epsilon}")
        if config.likelihood not in ['bernoulli', 'gaussian']:
            raise ValueError(f"likelihood must be 'bernoulli' or 'gaussian', got {config.likelihood}")

    def sample_target_classes(self, y_orig: torch.Tensor) -> torch.Tensor:
        """Sample target classes different from original classes.

        For each sample, randomly selects a different class.

        Args:
            y_orig: Original class labels [batch_size]

        Returns:
            Target class labels [batch_size]
        """
        batch_size = y_orig.size(0)
        num_classes = self.model.num_classes

        # Sample random classes
        y_target = torch.randint(0, num_classes - 1, (batch_size,), device=self.device)

        # Shift to ensure different from original
        # If sampled >= y_orig, increment by 1
        mask = y_target >= y_orig
        y_target[mask] += 1

        return y_target

    def compute_loss(
        self,
        x: torch.Tensor,
        y_orig: torch.Tensor,
        y_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute unified CF-TCVAE loss (Equation 15).

        L = recon + α·proximity + λ·validity + KL + (β-1)·TC + aux

        Args:
            x: Input samples [batch_size, *dims]
            y_orig: Original class labels [batch_size]
            y_target: Target class labels [batch_size]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Forward pass through encoder
        z_mean, z_logvar = self.model.encoder(x, y_target)

        # Reparameterization trick
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + std * eps

        # Update class-conditional priors via EMA (training mode only)
        if self.model.training:
            self.model.prior.update(z_mean, z_logvar, y_target)

        # Decode to reconstruction (original class)
        if self.config.likelihood == 'bernoulli':
            x_recon = self.model.decoder(z, y_orig)
            x_recon_logvar = None
        else:
            x_recon, x_recon_logvar = self.model.decoder(z, y_orig)

        # Decode to counterfactual (target class)
        if self.config.likelihood == 'bernoulli':
            x_cf = self.model.decoder(z, y_target)
            x_cf_logvar = None
        else:
            x_cf, x_cf_logvar = self.model.decoder(z, y_target)

        # 1. Reconstruction loss: -log P(x|z,y_orig)
        if self.config.likelihood == 'bernoulli':
            recon_loss = reconstruction_loss_bernoulli(x_recon, x, reduction='mean')
        else:
            recon_loss = reconstruction_loss_gaussian(x_recon, x_recon_logvar, x, reduction='mean')

        # 2. Proximity loss: α·||x - x_cf||₁
        proximity_loss = proximity_loss_l1(x_cf, x, self.config.alpha_proximity, reduction='mean')

        # 3. Validity hinge loss: λ·max(0, max_{y≠y'} f(x_cf)_y - f(x_cf)_{y'} + ε)
        with torch.no_grad():
            logits_cf = self.classifier(x_cf)
        validity_loss_value = hinge_loss_validity(
            logits_cf, y_target, self.config.epsilon, reduction='mean'
        )
        validity_loss = self.config.lambda_validity * validity_loss_value

        # Compute validity rate for monitoring
        validity_rate = compute_validity_rate(logits_cf, y_target)

        # 4. KL divergence: KL(Q(z|x,y')||P(z|y'))
        prior_mean, prior_logvar = self.model.prior.get_params(y_target)
        kl_loss = kl_divergence_diagonal_gaussian(
            z_mean, z_logvar, prior_mean, prior_logvar, reduction='mean'
        )

        # 5. Class-conditional TC: (β-1)·TC(z|y')
        tc_loss = self.model.tc_estimator.estimate_tc(z, y_target)

        # 6. Auxiliary classifier loss (with warmup)
        aux_loss = None
        if self.model.use_aux_classifier:
            aux_logits = self.model.auxiliary_classifier(z)
            aux_loss = auxiliary_classifier_loss(aux_logits, y_orig, reduction='mean')

        # Aggregate all losses
        total_loss, loss_dict = self.loss_aggregator.aggregate(
            recon_loss=recon_loss,
            proximity_loss=proximity_loss,
            validity_loss=validity_loss,
            kl_loss=kl_loss,
            tc_loss=tc_loss,
            aux_loss=aux_loss,
            current_epoch=self.current_epoch
        )

        # Add validity rate to loss dict
        loss_dict['validity_rate'] = validity_rate.item()

        return total_loss, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of averaged training metrics
        """
        self.model.train()

        # Metrics accumulator
        metrics_sum = {}
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 2:
                x, y_orig = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            x = x.to(self.device)
            y_orig = y_orig.to(self.device)

            # Sample target classes
            y_target = self.sample_target_classes(y_orig)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass and compute loss
            total_loss, loss_dict = self.compute_loss(x, y_orig, y_target)

            # Backward pass
            total_loss.backward()

            # Gradient clipping (optional, helps stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            for key, value in loss_dict.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            num_batches += 1

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'val_rate': f"{loss_dict['validity_rate']:.3f}"
                })

            self.global_step += 1

        # Average metrics
        metrics_avg = {k: v / num_batches for k, v in metrics_sum.items()}

        # Log epoch summary
        logger.info(
            f"Epoch {self.current_epoch} Training - "
            f"loss: {metrics_avg['total']:.4f}, "
            f"recon: {metrics_avg['recon']:.4f}, "
            f"validity: {metrics_avg['validity']:.4f}, "
            f"tc: {metrics_avg['tc']:.4f}, "
            f"validity_rate: {metrics_avg['validity_rate']:.3f}"
        )

        return metrics_avg

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        # Metrics accumulator
        metrics_sum = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            # Unpack batch
            if len(batch) == 2:
                x, y_orig = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            x = x.to(self.device)
            y_orig = y_orig.to(self.device)

            # Sample target classes
            y_target = self.sample_target_classes(y_orig)

            # Compute loss
            total_loss, loss_dict = self.compute_loss(x, y_orig, y_target)

            # Accumulate metrics
            for key, value in loss_dict.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            num_batches += 1

        # Average metrics
        metrics_avg = {f'val_{k}': v / num_batches for k, v in metrics_sum.items()}

        # Log validation summary
        logger.info(
            f"Epoch {self.current_epoch} Validation - "
            f"val_loss: {metrics_avg['val_total']:.4f}, "
            f"val_recon: {metrics_avg['val_recon']:.4f}, "
            f"val_validity_rate: {metrics_avg['val_validity_rate']:.3f}"
        )

        return metrics_avg

    def train(
        self,
        max_epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train CF-TCVAE for specified number of epochs.

        Args:
            max_epochs: Maximum epochs to train (uses config.max_epochs if None)
            early_stopping_patience: Stop if val_loss doesn't improve for N epochs (optional)

        Returns:
            Dictionary with training history and best metrics
        """
        if max_epochs is None:
            max_epochs = self.config.max_epochs

        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_validity_rate': [],
            'val_validity_rate': [],
            'lr': []
        }

        # Early stopping tracking
        epochs_without_improvement = 0

        start_time = time.time()

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate_epoch()

            # Learning rate scheduling
            val_loss = val_metrics['val_total']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_loss)
            history['train_validity_rate'].append(train_metrics['validity_rate'])
            history['val_validity_rate'].append(val_metrics['val_validity_rate'])
            history['lr'].append(current_lr)

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            if is_best or (epoch % 10 == 0):  # Save best + every 10 epochs
                metrics_dict = {**train_metrics, **val_metrics}
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics_dict,
                    is_best=is_best
                )

            # Log epoch time
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s, lr={current_lr:.6f}")

            # Early stopping check
            if early_stopping_patience is not None:
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered: no improvement for "
                        f"{early_stopping_patience} epochs"
                    )
                    break

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint_path()
        if best_checkpoint:
            logger.info(f"Loading best model from {best_checkpoint}")
            self.checkpoint_manager.load_checkpoint(
                best_checkpoint,
                model=self.model,
                device=self.device
            )

        return {
            'history': history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time
        }
