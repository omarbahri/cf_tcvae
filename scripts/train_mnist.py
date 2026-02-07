#!/usr/bin/env python3
"""Training script for CF-TCVAE on MNIST dataset.

Single-phase unified training with all loss components from Equation 15.
NO phase1/phase2 logic.

Usage:
    python scripts/train_mnist.py --max_epochs 150 --batch_size 1024 --beta 10.0
    python scripts/train_mnist.py --config configs/mnist_default.yaml
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cf_tcvae.config import MNISTConfig
from cf_tcvae.datasets import create_mnist_loaders
from cf_tcvae.classifiers import load_mnist_classifier
from cf_tcvae.models import CFTCVAE
from cf_tcvae.training import CFTCVAETrainer


def setup_logging(script_name: str, timestamp: str) -> logging.Logger:
    """Setup logging to file and console.

    Args:
        script_name: Name of the script
        timestamp: Timestamp string for log filename

    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path('logs') / script_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{script_name}_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CF-TCVAE on MNIST')

    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides other args)')

    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_patience', type=int, default=7,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR reduction factor')

    # Model hyperparameters
    parser.add_argument('--beta', type=float, default=10.0,
                        help='TC regularization weight')
    parser.add_argument('--alpha_proximity', type=float, default=300.0,
                        help='Proximity loss weight')
    parser.add_argument('--lambda_validity', type=float, default=1.0,
                        help='Validity loss weight')
    parser.add_argument('--epsilon', type=float, default=10.0,
                        help='Hinge loss margin')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--num_ch', type=int, default=32,
                        help='Base number of channels in CNN')

    # Auxiliary classifier
    parser.add_argument('--aux_weight', type=float, default=0.5,
                        help='Auxiliary classifier loss weight')
    parser.add_argument('--aux_warmup_epochs', type=int, default=10,
                        help='Epochs before auxiliary loss activates')

    # Prior
    parser.add_argument('--prior_ema_momentum', type=float, default=0.0001,
                        help='EMA momentum for prior updates')

    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/mnist',
                        help='Directory to save checkpoints')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (uses default if None)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience (disabled if None)')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
    logger = setup_logging('train_mnist', timestamp)

    logger.info("=" * 80)
    logger.info("CF-TCVAE Training on MNIST")
    logger.info("=" * 80)

    # Load configuration
    if args.config is not None:
        logger.info(f"Loading config from: {args.config}")
        config = MNISTConfig.from_yaml(args.config)
    else:
        logger.info("Using default MNIST config with CLI overrides")
        config = MNISTConfig()

        # Override with CLI arguments
        config.max_epochs = args.max_epochs
        config.batch_size = args.batch_size
        config.lr = args.lr
        config.lr_patience = args.lr_patience
        config.lr_factor = args.lr_factor
        config.beta = args.beta
        config.alpha_proximity = args.alpha_proximity
        config.lambda_validity = args.lambda_validity
        config.epsilon = args.epsilon
        config.z_dim = args.z_dim
        config.num_ch = args.num_ch
        config.aux_weight = args.aux_weight
        config.aux_warmup_epochs = args.aux_warmup_epochs
        config.prior_ema_momentum = args.prior_ema_momentum
        config.seed = args.seed

    # Log configuration
    logger.info("Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device

    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    logger.info("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = create_mnist_loaders(
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Load black-box classifier
    logger.info("Loading pre-trained MNIST classifier...")
    classifier = load_mnist_classifier(device=device)
    logger.info("Classifier loaded and frozen")

    # Create CF-TCVAE model
    logger.info("Creating CF-TCVAE model...")
    model = CFTCVAE(
        dataset='mnist',
        num_classes=10,
        z_dim=config.z_dim,
        num_ch=config.num_ch,
        class_emb_dim=config.class_emb_dim,
        likelihood='bernoulli',
        # Prior settings
        use_class_conditional_prior=config.use_class_conditional_prior,
        prior_ema_momentum=config.prior_ema_momentum,
        prior_init_std=config.prior_init_std,
        # TC settings
        use_class_conditional_tc=config.use_class_conditional_tc,
        tc_bank_capacity_per_class=config.bank_capacity_per_class,
        tc_min_samples_per_class=config.min_samples_per_class,
        # Auxiliary classifier
        use_aux_classifier=config.use_aux_classifier,
        aux_on_sampled_z=config.aux_on_sampled_z,
    )
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = CFTCVAETrainer(
        model=model,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        log_interval=50
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"Max epochs: {config.max_epochs}")
    logger.info(f"Early stopping patience: {args.early_stopping_patience}")

    results = trainer.train(
        max_epochs=config.max_epochs,
        early_stopping_patience=args.early_stopping_patience
    )

    # Log final results
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Total epochs: {results['total_epochs']}")
    logger.info(f"Total time: {results['total_time'] / 60:.1f} minutes")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Best checkpoint: {checkpoint_dir / 'best_model.pt'}")

    # Save final history
    history_file = checkpoint_dir / 'training_history.pt'
    torch.save(results['history'], history_file)
    logger.info(f"Training history saved to: {history_file}")

    logger.info("Training script completed successfully!")


if __name__ == '__main__':
    main()
