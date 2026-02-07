"""CF-TCVAE Model Architectures.

This package contains all model components for CF-TCVAE:
- Encoders: CNN (MNIST) and MLP (Adult) encoders
- Decoders: CNN (MNIST) and MLP (Adult) decoders
- TC Estimator: Class-conditional total correlation estimation
- Priors: Class-conditional Gaussian priors with EMA updates
- Auxiliary: Auxiliary classifier head with warmup
- CF-TCVAE: Main model with unified loss function

All components are FiLM-free and use concatenation for class conditioning.

Author: CF-TCVAE Team
Date: 2026-02-06
"""

from .encoders import MNISTEncoder, AdultEncoder, create_encoder
from .decoders import create_decoder
from .tc_estimator import MinibatchTCEstimator
from .priors import (
    ClassConditionalGaussianPrior,
    StandardGaussianPrior,
    create_prior
)
from .auxiliary import (
    AuxiliaryClassifier,
    AuxiliaryClassifierWithWarmup,
    create_auxiliary_classifier
)
from .cf_tcvae import CFTCVAE

__all__ = [
    # Encoders
    'MNISTEncoder',
    'AdultEncoder',
    'create_encoder',
    # Decoders
    'create_decoder',
    # TC Estimator
    'MinibatchTCEstimator',
    # Priors
    'ClassConditionalGaussianPrior',
    'StandardGaussianPrior',
    'create_prior',
    # Auxiliary
    'AuxiliaryClassifier',
    'AuxiliaryClassifierWithWarmup',
    'create_auxiliary_classifier',
    # Main model
    'CFTCVAE',
]
