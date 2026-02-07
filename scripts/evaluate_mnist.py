#!/usr/bin/env python3
"""Evaluation script for CF-TCVAE on MNIST dataset.

This script loads a trained CF-TCVAE model and evaluates it on the MNIST test set
using all 8 metrics from Table 1 of the paper.

Usage:
    python scripts/evaluate_mnist.py --checkpoint checkpoints/mnist_best.pth
    python scripts/evaluate_mnist.py --checkpoint checkpoints/mnist_best.pth --num_samples 1000
    python scripts/evaluate_mnist.py --checkpoint checkpoints/mnist_best.pth --save_visualizations
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cf_tcvae.config import MNISTConfig
from cf_tcvae.datasets import create_mnist_loaders
from cf_tcvae.classifiers import load_mnist_classifier
from cf_tcvae.models import CFTCVAE
from cf_tcvae.evaluation import (
    compute_validity_rate,
    compute_prediction_gain,
    compute_proximity_l1,
    compute_invalidation_rate,
    compute_robustness,
    compute_latency,
    train_global_autoencoder,
    train_class_autoencoders,
    compute_plausibility_ae,
    compute_plausibility_tc_ae,
)


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


def parse_timestamp_arg():
    """Parse timestamp from command line arguments for consistent log file naming."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--timestamp', type=str, help='Timestamp for log file naming')
    known_args, _ = parser.parse_known_args()
    return known_args.timestamp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CF-TCVAE on MNIST')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained CF-TCVAE checkpoint')

    # Evaluation parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for counterfactual generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--target_class', type=int, default=None,
                        help='Target class for counterfactuals (default: cycle through all)')

    # Autoencoder training for plausibility metrics
    parser.add_argument('--ae_epochs', type=int, default=50,
                        help='Epochs to train autoencoders for plausibility metrics')
    parser.add_argument('--ae_batch_size', type=int, default=128,
                        help='Batch size for autoencoder training')
    parser.add_argument('--skip_plausibility', action='store_true',
                        help='Skip plausibility metrics (saves time)')

    # Robustness parameters
    parser.add_argument('--robustness_sigma', type=float, default=0.05,
                        help='Noise level for robustness/invalidation rate')
    parser.add_argument('--robustness_samples', type=int, default=1,
                        help='Number of noise samples for robustness')

    # Output options
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save sample counterfactual visualizations')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of visualizations to save')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Optional timestamp for log file naming
    parser.add_argument('--timestamp', type=str, help='Timestamp for log file naming')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str, logger: logging.Logger) -> CFTCVAE:
    """Load CF-TCVAE model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        logger: Logger instance

    Returns:
        Loaded model in eval mode

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Extract config from checkpoint
    if 'config' not in checkpoint:
        raise RuntimeError("Checkpoint missing 'config' key. Invalid checkpoint format.")

    config = checkpoint['config']
    logger.info(f"Loaded config: dataset={config.dataset}, z_dim={config.z_dim}, beta={config.beta}")

    # Create model
    model = CFTCVAE(
        dataset=config.dataset,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        class_emb_dim=config.class_emb_dim,
        beta=config.beta,
        alpha_proximity=config.alpha_proximity,
        lambda_validity=config.lambda_validity,
        epsilon=config.epsilon,
        likelihood=config.likelihood,
        use_aux_classifier=config.use_aux_classifier,
        aux_weight=config.aux_weight,
        aux_warmup_epochs=config.aux_warmup_epochs,
        prior_ema_momentum=config.prior_ema_momentum,
        num_ch=getattr(config, 'num_ch', 32),
    ).to(device)

    # Load state dict
    if 'model_state_dict' not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict' key. Invalid checkpoint format.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    return model


def generate_counterfactuals(
    model: CFTCVAE,
    dataloader: DataLoader,
    target_class: int,
    device: str,
    logger: logging.Logger
) -> tuple:
    """Generate counterfactuals for test set.

    Args:
        model: Trained CF-TCVAE model
        dataloader: DataLoader for test data
        target_class: Target class for counterfactuals (or None for cycling)
        device: Device to use
        logger: Logger instance

    Returns:
        Tuple of (originals, counterfactuals, original_labels, target_labels)
    """
    model.eval()

    originals_list = []
    counterfactuals_list = []
    original_labels_list = []
    target_labels_list = []

    num_classes = 10  # MNIST has 10 classes

    logger.info("Generating counterfactuals...")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Generating CFs")):
            x = x.to(device)
            y = y.to(device)

            # Determine target classes
            if target_class is not None:
                # Use specified target class
                y_target = torch.full_like(y, target_class)
            else:
                # Cycle through all classes (excluding original)
                y_target = (y + 1) % num_classes

            # Generate counterfactuals
            x_cf = model.generate_counterfactual(x, y_target, deterministic=True)

            # Store results
            originals_list.append(x.cpu())
            counterfactuals_list.append(x_cf.cpu())
            original_labels_list.append(y.cpu())
            target_labels_list.append(y_target.cpu())

    # Concatenate all batches
    originals = torch.cat(originals_list, dim=0)
    counterfactuals = torch.cat(counterfactuals_list, dim=0)
    original_labels = torch.cat(original_labels_list, dim=0)
    target_labels = torch.cat(target_labels_list, dim=0)

    logger.info(f"Generated {len(originals)} counterfactuals")

    return originals, counterfactuals, original_labels, target_labels


def evaluate_metrics(
    originals: torch.Tensor,
    counterfactuals: torch.Tensor,
    original_labels: torch.Tensor,
    target_labels: torch.Tensor,
    model: CFTCVAE,
    classifier: torch.nn.Module,
    train_loader: DataLoader,
    device: str,
    args,
    logger: logging.Logger
) -> dict:
    """Compute all 8 evaluation metrics.

    Args:
        originals: Original test samples
        counterfactuals: Generated counterfactuals
        original_labels: Original class labels
        target_labels: Target class labels
        model: CF-TCVAE model
        classifier: Black-box classifier
        train_loader: Training data loader (for autoencoder training)
        device: Device to use
        args: Command line arguments
        logger: Logger instance

    Returns:
        Dictionary with all metric values
    """
    results = {}

    # Move data to device
    originals = originals.to(device)
    counterfactuals = counterfactuals.to(device)
    original_labels = original_labels.to(device)
    target_labels = target_labels.to(device)

    # 1. Validity Rate
    logger.info("Computing validity rate...")
    validity_rate = compute_validity_rate(counterfactuals, target_labels, classifier)
    results['validity_rate'] = validity_rate
    logger.info(f"Validity Rate: {validity_rate:.4f} ({validity_rate*100:.2f}%)")

    # 2. Prediction Gain
    logger.info("Computing prediction gain...")
    prediction_gain = compute_prediction_gain(originals, counterfactuals, target_labels, classifier)
    results['prediction_gain'] = prediction_gain
    logger.info(f"Prediction Gain: {prediction_gain:.4f}")

    # 3. Proximity (L1)
    logger.info("Computing proximity...")
    proximity = compute_proximity_l1(originals, counterfactuals)
    results['proximity_l1'] = proximity
    logger.info(f"Proximity (L1): {proximity:.4f}")

    # 4 & 5. Plausibility (AE and tc-AE)
    if not args.skip_plausibility:
        logger.info("Training autoencoders for plausibility metrics...")

        # Train global autoencoder
        logger.info("Training global autoencoder...")
        global_ae = train_global_autoencoder(
            train_loader=train_loader,
            dataset='mnist',
            device=device,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            logger=logger
        )

        # Compute plausibility (AE)
        logger.info("Computing plausibility (AE)...")
        plausibility_ae = compute_plausibility_ae(
            counterfactuals=counterfactuals,
            autoencoder=global_ae,
            likelihood='bernoulli',
            device=device
        )
        results['plausibility_ae'] = plausibility_ae
        logger.info(f"Plausibility (AE): {plausibility_ae:.4f}")

        # Train per-class autoencoders
        logger.info("Training per-class autoencoders...")
        class_aes = train_class_autoencoders(
            train_loader=train_loader,
            dataset='mnist',
            num_classes=10,
            device=device,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            logger=logger
        )

        # Compute plausibility (tc-AE)
        logger.info("Computing plausibility (tc-AE)...")
        plausibility_tc_ae = compute_plausibility_tc_ae(
            counterfactuals=counterfactuals,
            target_classes=target_labels,
            class_autoencoders=class_aes,
            likelihood='bernoulli',
            device=device
        )
        results['plausibility_tc_ae'] = plausibility_tc_ae
        logger.info(f"Plausibility (tc-AE): {plausibility_tc_ae:.4f}")
    else:
        logger.info("Skipping plausibility metrics (--skip_plausibility)")
        results['plausibility_ae'] = None
        results['plausibility_tc_ae'] = None

    # 6. Invalidation Rate
    logger.info("Computing invalidation rate...")
    invalidation_rate = compute_invalidation_rate(
        counterfactuals=counterfactuals,
        target_classes=target_labels,
        classifier=classifier,
        sigma=args.robustness_sigma,
        num_samples=args.robustness_samples,
        device=device
    )
    results['invalidation_rate'] = invalidation_rate
    logger.info(f"Invalidation Rate (σ={args.robustness_sigma}): {invalidation_rate:.4f} ({invalidation_rate*100:.2f}%)")

    # 7. Robustness
    robustness = compute_robustness(
        counterfactuals=counterfactuals,
        target_classes=target_labels,
        classifier=classifier,
        sigma=args.robustness_sigma,
        num_samples=args.robustness_samples,
        device=device
    )
    results['robustness'] = robustness
    logger.info(f"Robustness (1 - IR): {robustness:.4f} ({robustness*100:.2f}%)")

    # 8. Latency
    logger.info("Computing latency...")
    # Use a subset for latency measurement
    num_latency_samples = min(1000, len(originals))
    latency = compute_latency(
        model=model,
        test_samples=originals[:num_latency_samples],
        target_classes=target_labels[:num_latency_samples],
        device=device
    )
    results['latency'] = latency
    logger.info(f"Latency: {latency:.6f} seconds per counterfactual")

    return results


def save_results(results: dict, args, timestamp: str, logger: logging.Logger):
    """Save evaluation results to CSV and JSON.

    Args:
        results: Dictionary of metric values
        args: Command line arguments
        timestamp: Timestamp string
        logger: Logger instance
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / f'mnist_evaluation_{timestamp}.csv'
    df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to CSV: {csv_path.absolute()}")

    # Save to JSON with metadata
    json_path = output_dir / f'mnist_evaluation_{timestamp}.json'
    full_results = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'target_class': args.target_class,
        'robustness_sigma': args.robustness_sigma,
        'seed': args.seed,
        'metrics': results,
    }

    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Results saved to JSON: {json_path.absolute()}")


def save_visualizations(
    originals: torch.Tensor,
    counterfactuals: torch.Tensor,
    original_labels: torch.Tensor,
    target_labels: torch.Tensor,
    classifier: torch.nn.Module,
    args,
    timestamp: str,
    logger: logging.Logger
):
    """Save sample counterfactual visualizations.

    Args:
        originals: Original samples
        counterfactuals: Generated counterfactuals
        original_labels: Original labels
        target_labels: Target labels
        classifier: Black-box classifier
        args: Command line arguments
        timestamp: Timestamp string
        logger: Logger instance
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualizations")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select samples to visualize
    num_viz = min(args.num_visualizations, len(originals))
    indices = torch.randperm(len(originals))[:num_viz]

    # Create figure
    fig, axes = plt.subplots(num_viz, 3, figsize=(9, 3 * num_viz))
    if num_viz == 1:
        axes = axes.reshape(1, -1)

    classifier.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x_orig = originals[idx:idx+1].to(classifier.device if hasattr(classifier, 'device') else 'cpu')
            x_cf = counterfactuals[idx:idx+1].to(x_orig.device)

            # Get predictions
            pred_orig = classifier(x_orig).argmax(dim=1).item()
            pred_cf = classifier(x_cf).argmax(dim=1).item()

            y_orig = original_labels[idx].item()
            y_target = target_labels[idx].item()

            # Plot original
            axes[i, 0].imshow(x_orig.cpu().squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Original\nTrue: {y_orig}, Pred: {pred_orig}')
            axes[i, 0].axis('off')

            # Plot counterfactual
            axes[i, 1].imshow(x_cf.cpu().squeeze(), cmap='gray')
            axes[i, 1].set_title(f'Counterfactual\nTarget: {y_target}, Pred: {pred_cf}')
            axes[i, 1].axis('off')

            # Plot difference
            diff = (x_cf - x_orig).abs().cpu().squeeze()
            axes[i, 2].imshow(diff, cmap='hot')
            axes[i, 2].set_title(f'|Difference|\nL1: {diff.sum():.2f}')
            axes[i, 2].axis('off')

    plt.tight_layout()
    viz_path = output_dir / f'mnist_samples_{timestamp}.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizations saved to: {viz_path.absolute()}")


def print_summary(results: dict, logger: logging.Logger):
    """Print summary table of all metrics.

    Args:
        results: Dictionary of metric values
        logger: Logger instance
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY - MNIST CF-TCVAE")
    logger.info("="*60)
    logger.info(f"{'Metric':<30} {'Value':>15}")
    logger.info("-"*60)
    logger.info(f"{'Validity Rate (%):':<30} {results['validity_rate']*100:>14.2f}%")
    logger.info(f"{'Prediction Gain:':<30} {results['prediction_gain']:>15.4f}")
    logger.info(f"{'Proximity (L1):':<30} {results['proximity_l1']:>15.4f}")

    if results['plausibility_ae'] is not None:
        logger.info(f"{'Plausibility (AE):':<30} {results['plausibility_ae']:>15.4f}")
    else:
        logger.info(f"{'Plausibility (AE):':<30} {'N/A':>15}")

    if results['plausibility_tc_ae'] is not None:
        logger.info(f"{'Plausibility (tc-AE):':<30} {results['plausibility_tc_ae']:>15.4f}")
    else:
        logger.info(f"{'Plausibility (tc-AE):':<30} {'N/A':>15}")

    logger.info(f"{'Invalidation Rate (%):':<30} {results['invalidation_rate']*100:>14.2f}%")
    logger.info(f"{'Robustness (%):':<30} {results['robustness']*100:>14.2f}%")
    logger.info(f"{'Latency (s):':<30} {results['latency']:>15.6f}")
    logger.info("="*60 + "\n")


def main():
    """Main evaluation function."""
    # Parse timestamp first
    timestamp = parse_timestamp_arg()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

    # Setup logging
    logger = setup_logging('evaluate_mnist', timestamp)

    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("="*60)
    logger.info("CF-TCVAE MNIST EVALUATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Number of samples: {args.num_samples or 'all'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Target class: {args.target_class or 'cycle through all'}")
    logger.info("="*60 + "\n")

    # Load datasets
    logger.info("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = create_mnist_loaders(
        batch_size=args.batch_size,
        test_batch_size=args.batch_size
    )

    # Subset test set if requested
    if args.num_samples is not None:
        test_dataset = test_loader.dataset
        indices = list(range(min(args.num_samples, len(test_dataset))))
        test_subset = Subset(test_dataset, indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        logger.info(f"Using {len(test_subset)} test samples")

    # Load classifier
    logger.info("Loading black-box classifier...")
    classifier = load_mnist_classifier(device=args.device)
    logger.info("Classifier loaded successfully")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device, logger)

    # Generate counterfactuals
    originals, counterfactuals, original_labels, target_labels = generate_counterfactuals(
        model=model,
        dataloader=test_loader,
        target_class=args.target_class,
        device=args.device,
        logger=logger
    )

    # Evaluate metrics
    results = evaluate_metrics(
        originals=originals,
        counterfactuals=counterfactuals,
        original_labels=original_labels,
        target_labels=target_labels,
        model=model,
        classifier=classifier,
        train_loader=train_loader,
        device=args.device,
        args=args,
        logger=logger
    )

    # Print summary
    print_summary(results, logger)

    # Save results
    save_results(results, args, timestamp, logger)

    # Save visualizations if requested
    if args.save_visualizations:
        logger.info("Generating visualizations...")
        save_visualizations(
            originals=originals,
            counterfactuals=counterfactuals,
            original_labels=original_labels,
            target_labels=target_labels,
            classifier=classifier,
            args=args,
            timestamp=timestamp,
            logger=logger
        )

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
