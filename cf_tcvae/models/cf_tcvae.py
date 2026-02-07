"""CF-TCVAE: Counterfactual Total Correlation VAE.

This module implements the main CF-TCVAE model with unified single-phase training.
The model generates counterfactual explanations by encoding inputs conditioned on
target classes and decoding to produce counterfactuals.

Loss function (Equation 15 from paper):
    L = E_Q(z|x,y')[log P(x|z,y') + α·||x-x_cf||₁ + λ·HingeLoss(f(x_cf),y',ε)]
        + (β-1)·TC(z|y') + KL(Q(z|x,y')||P(z|y')) + L_aux(z,y)

Where:
    - P(x|z,y') is the reconstruction likelihood (Bernoulli or Gaussian)
    - α is the proximity weight
    - λ is the validity weight
    - ε is the hinge loss margin
    - β is the TC regularization weight
    - TC(z|y') is the class-conditional total correlation
    - P(z|y') is the class-conditional prior
    - L_aux is the auxiliary classifier loss

Author: CF-TCVAE Team
Date: 2026-02-06
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders import create_encoder
from .decoders import create_decoder
from .tc_estimator import MinibatchTCEstimator
from .priors import create_prior
from .auxiliary import create_auxiliary_classifier


class CFTCVAE(nn.Module):
    """Counterfactual Total Correlation VAE.

    Unified single-phase training model that generates counterfactual explanations
    by conditioning on target classes.

    Args:
        dataset: Dataset name ('mnist' or 'adult')
        z_dim: Latent dimension
        num_classes: Number of classes
        class_emb_dim: Class embedding dimension
        beta: TC regularization weight
        alpha_proximity: Proximity loss weight
        lambda_validity: Validity loss weight
        epsilon: Hinge loss margin
        likelihood: Reconstruction likelihood ('bernoulli' or 'gaussian')
        use_class_conditional_prior: Whether to use class-conditional prior
        prior_ema_momentum: EMA momentum for prior updates
        prior_init_std: Initial prior standard deviation
        use_class_conditional_tc: Whether to use class-conditional TC
        min_samples_per_class: Min samples per class for TC estimation
        bank_capacity_per_class: Memory bank capacity per class
        use_aux_classifier: Whether to use auxiliary classifier
        aux_weight: Auxiliary classifier loss weight
        aux_warmup_epochs: Warmup epochs for auxiliary classifier
        **kwargs: Dataset-specific arguments (e.g., num_ch for MNIST, input_dim for Adult)
    """

    def __init__(
        self,
        dataset: str,
        z_dim: int,
        num_classes: int,
        class_emb_dim: int = 32,
        beta: float = 10.0,
        alpha_proximity: float = 1.0,
        lambda_validity: float = 1.0,
        epsilon: float = 5.0,
        likelihood: str = 'bernoulli',
        use_class_conditional_prior: bool = True,
        prior_ema_momentum: float = 0.01,
        prior_init_std: float = 1.0,
        use_class_conditional_tc: bool = True,
        min_samples_per_class: int = 8,
        bank_capacity_per_class: int = 256,
        use_aux_classifier: bool = True,
        aux_weight: float = 0.5,
        aux_warmup_epochs: int = 10,
        **kwargs
    ):
        super().__init__()

        # Validate arguments (NO FALLBACKS)
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if class_emb_dim < 0:
            raise ValueError(f"class_emb_dim must be non-negative, got {class_emb_dim}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if alpha_proximity < 0:
            raise ValueError(f"alpha_proximity must be non-negative, got {alpha_proximity}")
        if lambda_validity < 0:
            raise ValueError(f"lambda_validity must be non-negative, got {lambda_validity}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if likelihood not in ['bernoulli', 'gaussian']:
            raise ValueError(
                f"likelihood must be 'bernoulli' or 'gaussian', got '{likelihood}'. "
                "No fallback will be used."
            )

        self.dataset = dataset.lower()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.beta = beta
        self.alpha_proximity = alpha_proximity
        self.lambda_validity = lambda_validity
        self.epsilon = epsilon
        self.likelihood = likelihood

        # Create class embedding layer
        if class_emb_dim > 0:
            self.class_embedding = nn.Embedding(num_classes, class_emb_dim)
        else:
            self.class_embedding = None

        # Create encoder and decoder
        self.encoder = create_encoder(
            dataset=dataset,
            z_dim=z_dim,
            class_emb_dim=class_emb_dim,
            **kwargs
        )
        self.decoder = create_decoder(
            dataset=dataset,
            z_dim=z_dim,
            class_emb_dim=class_emb_dim,
            **kwargs
        )

        # Create class-conditional prior
        self.prior = create_prior(
            latent_dim=z_dim,
            num_classes=num_classes,
            use_class_conditional=use_class_conditional_prior,
            init_std=prior_init_std,
            ema_momentum=prior_ema_momentum
        )

        # Create TC estimator
        self.tc_estimator = MinibatchTCEstimator(
            latent_dim=z_dim,
            num_classes=num_classes,
            use_class_conditional=use_class_conditional_tc,
            min_samples_per_class=min_samples_per_class,
            bank_capacity_per_class=bank_capacity_per_class
        )

        # Create auxiliary classifier (optional)
        if use_aux_classifier:
            self.aux_classifier = create_auxiliary_classifier(
                latent_dim=z_dim,
                num_classes=num_classes,
                use_warmup=True,
                warmup_epochs=aux_warmup_epochs,
                weight=aux_weight
            )
        else:
            self.aux_classifier = None

    def encode(self, x: torch.Tensor, target_class: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input conditioned on target class.

        Args:
            x: Input data [batch_size, ...]
            target_class: Target class labels [batch_size]

        Returns:
            z_mean: Latent mean [batch_size, z_dim]
            z_logvar: Latent log-variance [batch_size, z_dim]
        """
        # Get class embeddings if using class conditioning
        if self.class_embedding is not None:
            class_emb = self.class_embedding(target_class)
        else:
            class_emb = None

        # Encode
        z_mean, z_logvar = self.encoder(x, class_emb)

        return z_mean, z_logvar

    def decode(self, z: torch.Tensor, target_class: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent code conditioned on target class.

        Args:
            z: Latent code [batch_size, z_dim]
            target_class: Target class labels [batch_size]

        Returns:
            x_mean: Reconstruction mean [batch_size, ...]
            x_logvar: Reconstruction log-variance [batch_size, ...]
        """
        # Get class embeddings if using class conditioning
        if self.class_embedding is not None:
            class_emb = self.class_embedding(target_class)
        else:
            class_emb = None

        # Decode
        x_mean, x_logvar = self.decoder(z, class_emb)

        return x_mean, x_logvar

    def reparameterize(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,I).

        Args:
            z_mean: Latent mean [batch_size, z_dim]
            z_logvar: Latent log-variance [batch_size, z_dim]

        Returns:
            z: Sampled latent code [batch_size, z_dim]
        """
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z

    def forward(
        self,
        x: torch.Tensor,
        original_class: torch.Tensor,
        target_class: torch.Tensor,
        classifier: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through CF-TCVAE.

        Args:
            x: Input data [batch_size, ...]
            original_class: Original class labels [batch_size]
            target_class: Target class labels [batch_size]
            classifier: Black-box classifier for validity loss (optional)

        Returns:
            x_cf: Counterfactual reconstructions [batch_size, ...]
            losses: Dictionary of loss components
        """
        # Encode conditioned on target class
        z_mean, z_logvar = self.encode(x, target_class)

        # Sample latent code
        z = self.reparameterize(z_mean, z_logvar)

        # Decode to counterfactual
        x_cf, x_cf_logvar = self.decode(z, target_class)

        # Compute loss components
        losses = self.compute_loss(
            x=x,
            x_cf=x_cf,
            x_cf_logvar=x_cf_logvar,
            z=z,
            z_mean=z_mean,
            z_logvar=z_logvar,
            original_class=original_class,
            target_class=target_class,
            classifier=classifier
        )

        # Update priors (EMA)
        if self.training:
            self.prior.update_priors(z_mean, z_logvar, target_class)

        return x_cf, losses

    def compute_loss(
        self,
        x: torch.Tensor,
        x_cf: torch.Tensor,
        x_cf_logvar: torch.Tensor,
        z: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        original_class: torch.Tensor,
        target_class: torch.Tensor,
        classifier: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components (Equation 15 from paper).

        Args:
            x: Input data [batch_size, ...]
            x_cf: Counterfactual reconstructions [batch_size, ...]
            x_cf_logvar: CF log-variance [batch_size, ...]
            z: Sampled latent codes [batch_size, z_dim]
            z_mean: Latent means [batch_size, z_dim]
            z_logvar: Latent log-variances [batch_size, z_dim]
            original_class: Original class labels [batch_size]
            target_class: Target class labels [batch_size]
            classifier: Black-box classifier for validity loss

        Returns:
            Dictionary with loss components and total loss
        """
        batch_size = x.size(0)

        # 1. Reconstruction likelihood: -log P(x|z,y')
        if self.likelihood == 'bernoulli':
            # Binary cross-entropy for images
            recon_loss = F.binary_cross_entropy(x_cf, x, reduction='none')
            recon_loss = recon_loss.view(batch_size, -1).sum(dim=1)
        elif self.likelihood == 'gaussian':
            # Gaussian likelihood for tabular data
            # -log N(x; x_cf, exp(x_cf_logvar)) = 0.5 * [log(2π) + x_cf_logvar + (x-x_cf)²/exp(x_cf_logvar)]
            recon_loss = 0.5 * (
                math.log(2 * math.pi) +
                x_cf_logvar +
                (x - x_cf) ** 2 / torch.exp(x_cf_logvar).clamp(min=1e-6)
            )
            recon_loss = recon_loss.view(batch_size, -1).sum(dim=1)
        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        # 2. Proximity loss: α·||x - x_cf||₁
        proximity_loss = torch.abs(x - x_cf).view(batch_size, -1).sum(dim=1)
        proximity_loss = self.alpha_proximity * proximity_loss

        # 3. Validity loss: λ·HingeLoss(f(x_cf), y', ε)
        if classifier is not None and self.lambda_validity > 0:
            with torch.no_grad():
                logits_cf = classifier(x_cf)
            validity_loss = self.hinge_loss(logits_cf, target_class, self.epsilon)
            validity_loss = self.lambda_validity * validity_loss
        else:
            validity_loss = torch.zeros(batch_size, device=x.device)

        # 4. KL divergence: KL(Q(z|x,y')||P(z|y'))
        kl_loss = self.prior.kl_divergence(z_mean, z_logvar, target_class)

        # 5. Total Correlation: (β-1)·TC(z|y')
        tc_loss_value = self.tc_estimator(z, target_class, z_mean, z_logvar)
        tc_loss = (self.beta - 1.0) * tc_loss_value

        # 6. Auxiliary classifier loss: L_aux(z, y)
        if self.aux_classifier is not None:
            aux_loss = self.aux_classifier.compute_loss(z, original_class, reduction='none')
        else:
            aux_loss = torch.zeros(batch_size, device=x.device)

        # Total loss (average over batch)
        total_loss = (
            recon_loss.mean() +
            proximity_loss.mean() +
            validity_loss.mean() +
            kl_loss.mean() +
            tc_loss +  # TC is already averaged in tc_estimator
            aux_loss.mean()
        )

        # Return all components
        return {
            'total': total_loss,
            'recon': recon_loss.mean(),
            'proximity': proximity_loss.mean(),
            'validity': validity_loss.mean(),
            'kl': kl_loss.mean(),
            'tc': tc_loss,
            'aux': aux_loss.mean(),
        }

    def hinge_loss(
        self,
        logits: torch.Tensor,
        target_class: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Hinge loss ensuring counterfactual is classified as target with margin ε.

        L = max(0, max_{y≠y'} f(x_cf)_y - f(x_cf)_{y'} + ε)

        Args:
            logits: Classifier logits [batch_size, num_classes]
            target_class: Target class labels [batch_size]
            epsilon: Hinge margin

        Returns:
            Hinge loss per sample [batch_size]
        """
        batch_size = logits.size(0)

        # Get probability for target class
        probs = F.softmax(logits, dim=1)
        target_probs = probs.gather(1, target_class.unsqueeze(1))  # [batch, 1]

        # Get max probability for non-target classes
        probs_copy = probs.clone()
        probs_copy.scatter_(1, target_class.unsqueeze(1), -1e9)  # Mask target class
        max_other_probs = probs_copy.max(dim=1, keepdim=True)[0]  # [batch, 1]

        # Hinge loss
        loss = torch.clamp(max_other_probs - target_probs + epsilon, min=0.0)

        return loss.squeeze()

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Generate counterfactual explanation for given input and target class.

        Args:
            x: Input data [batch_size, ...]
            target_class: Target class labels [batch_size]
            deterministic: If True, use mean of latent distribution (no sampling)

        Returns:
            x_cf: Counterfactual explanations [batch_size, ...]
        """
        with torch.no_grad():
            # Encode
            z_mean, z_logvar = self.encode(x, target_class)

            # Sample or use mean
            if deterministic:
                z = z_mean
            else:
                z = self.reparameterize(z_mean, z_logvar)

            # Decode
            x_cf, _ = self.decode(z, target_class)

        return x_cf

    def step_epoch(self):
        """Increment epoch counter for warmup schedules."""
        if self.aux_classifier is not None:
            self.aux_classifier.step_epoch()
