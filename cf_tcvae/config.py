"""
Configuration system for CF-TCVAE.

This module provides dataset-specific configuration dataclasses with validation
and YAML loading support. Configurations include default hyperparameters from
the best-performing checkpoints used in the paper.

CRITICAL: NO FALLBACKS POLICY
- All required parameters must be explicitly provided
- Invalid values raise clear errors (ValueError, TypeError)
- Missing required parameters raise ValueError
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any
import yaml
import os


@dataclass
class CFTCVAEConfig:
    """
    Base configuration for CF-TCVAE model and training.

    This class defines all required and optional parameters for training
    CF-TCVAE models. Dataset-specific configurations (MNISTConfig, AdultConfig)
    inherit from this and provide sensible defaults.

    IMPORTANT: When creating configurations from scratch (not using defaults),
    all required parameters must be explicitly provided.
    """

    # Core CF-TCVAE parameters (REQUIRED)
    beta: float  # TC regularization weight
    lr: float  # Learning rate
    epsilon: float  # Hinge loss margin for validity
    lambda_validity: float  # Validity loss weight
    alpha_proximity: float  # Proximity loss weight

    # Architecture (REQUIRED)
    z_dim: int  # Latent dimension
    class_emb_dim: int  # Class embedding dimension

    # Training parameters (REQUIRED)
    batch_size: int
    test_batch_size: int
    max_epochs: int
    lr_patience: int  # Learning rate scheduler patience
    lr_factor: float  # Learning rate reduction factor

    # Prior configuration (REQUIRED)
    prior: Literal['standard'] = 'standard'
    use_class_conditional_prior: bool = True
    prior_update_mode: Literal['ema', 'none'] = 'ema'
    prior_ema_momentum: float = field(default=0.0)  # Must be set explicitly per dataset
    prior_init_std: float = 1.0

    # Auxiliary Classifier (REQUIRED)
    use_aux_classifier: bool = True
    aux_weight: float = 0.5
    aux_warmup_epochs: int = 10
    aux_on_sampled_z: bool = False

    # Likelihood and reconstruction (REQUIRED)
    likelihood: Literal['bernoulli', 'gaussian'] = field(default='bernoulli')
    use_reconstruction_path: bool = True
    condition_on_target: bool = True

    # Optimization
    balance_gradients: bool = False
    gradient_balance_method: Literal['normalize', 'none'] = 'normalize'

    # Black-box classifier (REQUIRED)
    freeze_classifier: bool = True
    classifier_checkpoint: str = field(default='')  # Must be set per dataset

    # Reproducibility
    seed: int = 42

    # Dataset-specific (set by subclasses)
    dataset: str = field(default='')

    # Device
    device: str = 'cuda'
    num_workers: int = 4

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """
        Validate all configuration parameters.

        Raises:
            ValueError: If any parameter has an invalid value
            TypeError: If any parameter has an invalid type
        """
        # Validate beta
        if self.beta is None:
            raise ValueError("config.beta is required but not provided. Set beta value explicitly.")
        if not isinstance(self.beta, (int, float)):
            raise TypeError(f"config.beta must be a number, got {type(self.beta).__name__}")
        if self.beta < 0:
            raise ValueError(f"config.beta must be non-negative, got {self.beta}")

        # Validate learning rate
        if self.lr is None:
            raise ValueError("config.lr is required but not provided. Set learning rate explicitly.")
        if not isinstance(self.lr, (int, float)):
            raise TypeError(f"config.lr must be a number, got {type(self.lr).__name__}")
        if self.lr <= 0:
            raise ValueError(f"config.lr must be positive, got {self.lr}")

        # Validate epsilon
        if self.epsilon is None:
            raise ValueError("config.epsilon is required but not provided. Set hinge loss margin explicitly.")
        if not isinstance(self.epsilon, (int, float)):
            raise TypeError(f"config.epsilon must be a number, got {type(self.epsilon).__name__}")
        if self.epsilon < 0:
            raise ValueError(f"config.epsilon must be non-negative, got {self.epsilon}")

        # Validate lambda_validity
        if self.lambda_validity is None:
            raise ValueError("config.lambda_validity is required but not provided.")
        if not isinstance(self.lambda_validity, (int, float)):
            raise TypeError(f"config.lambda_validity must be a number, got {type(self.lambda_validity).__name__}")
        if self.lambda_validity < 0:
            raise ValueError(f"config.lambda_validity must be non-negative, got {self.lambda_validity}")

        # Validate alpha_proximity
        if self.alpha_proximity is None:
            raise ValueError("config.alpha_proximity is required but not provided.")
        if not isinstance(self.alpha_proximity, (int, float)):
            raise TypeError(f"config.alpha_proximity must be a number, got {type(self.alpha_proximity).__name__}")
        if self.alpha_proximity < 0:
            raise ValueError(f"config.alpha_proximity must be non-negative, got {self.alpha_proximity}")

        # Validate z_dim
        if self.z_dim is None:
            raise ValueError("config.z_dim is required but not provided.")
        if not isinstance(self.z_dim, int):
            raise TypeError(f"config.z_dim must be an integer, got {type(self.z_dim).__name__}")
        if self.z_dim <= 0:
            raise ValueError(f"config.z_dim must be positive, got {self.z_dim}")

        # Validate class_emb_dim
        if self.class_emb_dim is None:
            raise ValueError("config.class_emb_dim is required but not provided.")
        if not isinstance(self.class_emb_dim, int):
            raise TypeError(f"config.class_emb_dim must be an integer, got {type(self.class_emb_dim).__name__}")
        if self.class_emb_dim <= 0:
            raise ValueError(f"config.class_emb_dim must be positive, got {self.class_emb_dim}")

        # Validate batch sizes
        if self.batch_size is None:
            raise ValueError("config.batch_size is required but not provided.")
        if not isinstance(self.batch_size, int):
            raise TypeError(f"config.batch_size must be an integer, got {type(self.batch_size).__name__}")
        if self.batch_size <= 0:
            raise ValueError(f"config.batch_size must be positive, got {self.batch_size}")

        if self.test_batch_size is None:
            raise ValueError("config.test_batch_size is required but not provided.")
        if not isinstance(self.test_batch_size, int):
            raise TypeError(f"config.test_batch_size must be an integer, got {type(self.test_batch_size).__name__}")
        if self.test_batch_size <= 0:
            raise ValueError(f"config.test_batch_size must be positive, got {self.test_batch_size}")

        # Validate max_epochs
        if self.max_epochs is None:
            raise ValueError("config.max_epochs is required but not provided.")
        if not isinstance(self.max_epochs, int):
            raise TypeError(f"config.max_epochs must be an integer, got {type(self.max_epochs).__name__}")
        if self.max_epochs <= 0:
            raise ValueError(f"config.max_epochs must be positive, got {self.max_epochs}")

        # Validate lr_patience
        if self.lr_patience is None:
            raise ValueError("config.lr_patience is required but not provided.")
        if not isinstance(self.lr_patience, int):
            raise TypeError(f"config.lr_patience must be an integer, got {type(self.lr_patience).__name__}")
        if self.lr_patience <= 0:
            raise ValueError(f"config.lr_patience must be positive, got {self.lr_patience}")

        # Validate lr_factor
        if self.lr_factor is None:
            raise ValueError("config.lr_factor is required but not provided.")
        if not isinstance(self.lr_factor, (int, float)):
            raise TypeError(f"config.lr_factor must be a number, got {type(self.lr_factor).__name__}")
        if self.lr_factor <= 0 or self.lr_factor >= 1:
            raise ValueError(f"config.lr_factor must be in (0, 1), got {self.lr_factor}")

        # Validate prior_ema_momentum
        if self.prior_update_mode == 'ema':
            if self.prior_ema_momentum is None or self.prior_ema_momentum == 0.0:
                raise ValueError(
                    "config.prior_ema_momentum must be set explicitly when prior_update_mode='ema'. "
                    "Use 0.0001 for MNIST or 0.001 for Adult datasets."
                )
            if not isinstance(self.prior_ema_momentum, (int, float)):
                raise TypeError(f"config.prior_ema_momentum must be a number, got {type(self.prior_ema_momentum).__name__}")
            if self.prior_ema_momentum < 0 or self.prior_ema_momentum > 1:
                raise ValueError(f"config.prior_ema_momentum must be in [0, 1], got {self.prior_ema_momentum}")

        # Validate aux_weight
        if self.aux_weight is None:
            raise ValueError("config.aux_weight is required but not provided.")
        if not isinstance(self.aux_weight, (int, float)):
            raise TypeError(f"config.aux_weight must be a number, got {type(self.aux_weight).__name__}")
        if self.aux_weight < 0:
            raise ValueError(f"config.aux_weight must be non-negative, got {self.aux_weight}")

        # Validate aux_warmup_epochs
        if self.aux_warmup_epochs is None:
            raise ValueError("config.aux_warmup_epochs is required but not provided.")
        if not isinstance(self.aux_warmup_epochs, int):
            raise TypeError(f"config.aux_warmup_epochs must be an integer, got {type(self.aux_warmup_epochs).__name__}")
        if self.aux_warmup_epochs < 0:
            raise ValueError(f"config.aux_warmup_epochs must be non-negative, got {self.aux_warmup_epochs}")

        # Validate likelihood
        if self.likelihood not in ['bernoulli', 'gaussian']:
            raise ValueError(f"config.likelihood must be 'bernoulli' or 'gaussian', got '{self.likelihood}'")

        # Validate classifier checkpoint
        if not self.classifier_checkpoint:
            raise ValueError(
                "config.classifier_checkpoint is required but not provided. "
                "Set the path to the pre-trained classifier checkpoint."
            )

        # Validate dataset
        if not self.dataset:
            raise ValueError(
                "config.dataset is required but not provided. "
                "Use 'mnist' or 'adult'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save_yaml(self, filepath: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: str):
        """Load config from YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Config file is empty: {filepath}")

        return cls(**config_dict)


@dataclass
class MNISTConfig(CFTCVAEConfig):
    """
    Default configuration for CF-TCVAE on MNIST dataset.

    These hyperparameters are extracted from the best-performing checkpoint
    (run_id: ovp4bhho) from the original experiments.

    Source: wandb/run-20250925_035048-ovp4bhho/files/config.yaml
    """

    # Core CF-TCVAE parameters
    beta: float = 10.0
    lr: float = 0.001
    epsilon: float = 10.0
    lambda_validity: float = 1.0
    alpha_proximity: float = 300.0

    # Architecture
    z_dim: int = 64
    num_ch: int = 32  # Number of channels in conv layers
    class_emb_dim: int = 32

    # Training
    batch_size: int = 1024
    test_batch_size: int = 1024
    max_epochs: int = 150
    lr_patience: int = 7
    lr_factor: float = 0.5

    # Prior
    prior: Literal['standard'] = 'standard'
    use_class_conditional_prior: bool = True
    prior_update_mode: Literal['ema'] = 'ema'
    prior_ema_momentum: float = 0.0001
    prior_init_std: float = 1.0

    # Total Correlation (class-conditional)
    use_class_conditional_tc: bool = True
    min_samples_per_class: int = 16
    bank_capacity_per_class: int = 1024

    # Auxiliary Classifier
    use_aux_classifier: bool = True
    aux_weight: float = 0.5
    aux_warmup_epochs: int = 10
    aux_on_sampled_z: bool = False

    # Likelihood and reconstruction
    likelihood: Literal['bernoulli'] = 'bernoulli'
    use_reconstruction_path: bool = True
    condition_on_target: bool = True

    # Optimization
    balance_gradients: bool = False
    gradient_balance_method: Literal['normalize'] = 'normalize'

    # Black-box classifier
    freeze_classifier: bool = True
    classifier_checkpoint: str = 'cf_tcvae/classifiers/checkpoints/mnist_advanced_best.pth'

    # Dataset
    dataset: str = 'mnist'

    # Reproducibility
    seed: int = 42


@dataclass
class AdultConfig(CFTCVAEConfig):
    """
    Default configuration for CF-TCVAE on Adult Income dataset.

    These hyperparameters are extracted from the best-performing checkpoint
    (run_id: 3fu93r06) from the original experiments.

    Key differences from MNIST:
    - Higher learning rate (0.005 vs 0.001)
    - Lower epsilon (5.0 vs 10.0)
    - Much lower alpha_proximity (2.0 vs 300.0) for tabular data
    - Smaller latent dimension (16 vs 64)
    - Gaussian likelihood (continuous features)
    - Higher prior_ema_momentum (0.001 vs 0.0001)

    Source: wandb/run-20251001_200008-3fu93r06/files/config.yaml
    """

    # Core CF-TCVAE parameters
    beta: float = 10.0
    lr: float = 0.005  # Higher than MNIST
    epsilon: float = 5.0  # Lower than MNIST
    lambda_validity: float = 1.0
    alpha_proximity: float = 2.0  # Much lower for tabular data

    # Architecture
    z_dim: int = 16  # Smaller for tabular data
    input_dim: int = 108  # Adult preprocessed feature dimension
    class_emb_dim: int = 32

    # Training
    batch_size: int = 1024
    test_batch_size: int = 1024
    max_epochs: int = 200
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Prior
    prior: Literal['standard'] = 'standard'
    use_class_conditional_prior: bool = True
    prior_update_mode: Literal['ema'] = 'ema'
    prior_ema_momentum: float = 0.001  # Higher than MNIST
    prior_init_std: float = 1.0

    # Auxiliary Classifier
    use_aux_classifier: bool = True
    aux_weight: float = 0.5
    aux_warmup_epochs: int = 10
    aux_on_sampled_z: bool = False

    # Likelihood and reconstruction
    likelihood: Literal['gaussian'] = 'gaussian'  # Gaussian for tabular data
    use_reconstruction_path: bool = True
    condition_on_target: bool = True
    use_all_target_classes: bool = True

    # Optimization
    balance_gradients: bool = False
    gradient_balance_method: Literal['normalize'] = 'normalize'

    # Black-box classifier
    freeze_classifier: bool = True
    classifier_checkpoint: str = 'cf_tcvae/classifiers/checkpoints/adult_mlp_full.pth'

    # Dataset
    dataset: str = 'adult'

    # Reproducibility
    seed: int = 42


def get_config(dataset: str, **overrides) -> CFTCVAEConfig:
    """
    Get default configuration for a dataset with optional overrides.

    Args:
        dataset: Dataset name ('mnist' or 'adult')
        **overrides: Optional parameter overrides

    Returns:
        Configuration object with defaults and overrides applied

    Raises:
        ValueError: If dataset is not supported

    Example:
        >>> config = get_config('mnist', beta=15.0, lr=0.002)
        >>> config.beta
        15.0
    """
    if dataset.lower() == 'mnist':
        config_dict = asdict(MNISTConfig())
    elif dataset.lower() == 'adult':
        config_dict = asdict(AdultConfig())
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. "
            f"Supported datasets: 'mnist', 'adult'"
        )

    # Apply overrides
    config_dict.update(overrides)

    # Create appropriate config class
    if dataset.lower() == 'mnist':
        return MNISTConfig(**config_dict)
    else:
        return AdultConfig(**config_dict)


def load_config(filepath: str, **overrides) -> CFTCVAEConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        filepath: Path to YAML config file
        **overrides: Optional parameter overrides

    Returns:
        Configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or dataset is unknown

    Example:
        >>> config = load_config('configs/mnist_default.yaml', beta=15.0)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Config file is empty: {filepath}")

    # Apply overrides
    config_dict.update(overrides)

    # Determine dataset and create appropriate config
    dataset = config_dict.get('dataset', '')
    if not dataset:
        raise ValueError(
            f"Config file must specify 'dataset' field. "
            f"Valid values: 'mnist', 'adult'"
        )

    if dataset.lower() == 'mnist':
        return MNISTConfig(**config_dict)
    elif dataset.lower() == 'adult':
        return AdultConfig(**config_dict)
    else:
        raise ValueError(
            f"Unknown dataset in config: '{dataset}'. "
            f"Supported datasets: 'mnist', 'adult'"
        )
