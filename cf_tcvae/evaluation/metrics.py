"""
Evaluation metrics for counterfactual explanations.

Implements 8 metrics from Table 1 of the CF-TCVAE paper:
1. Validity Rate (%): Percentage of counterfactuals where f(x_cf) = y'
2. Prediction Gain: Average confidence increase f(x_cf)_{y'} - f(x)_{y'}
3. Proximity (L1): Average L1 distance ||x - x_cf||₁
4. Plausibility (AE): Reconstruction error using global autoencoder
5. Plausibility (tc-AE): Reconstruction error using per-class autoencoders
6. Invalidation Rate (IR): Percentage flipping back when Gaussian noise σ=0.05 added
7. Robustness: 1 - IR (percentage remaining valid under noise)
8. Latency (s): Average generation time per counterfactual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Callable, Union

logger = logging.getLogger(__name__)


# ==================== 1. VALIDITY RATE ====================

def compute_validity_rate(
    counterfactuals: torch.Tensor,
    target_classes: torch.Tensor,
    classifier: nn.Module
) -> float:
    """
    Compute Validity Rate: percentage of counterfactuals correctly classified as target class.

    Validity = (1/N) Σ 1[f(x_cf) = y']

    Args:
        counterfactuals: Generated counterfactuals [N, *input_shape]
        target_classes: Target class labels [N]
        classifier: Trained classifier model

    Returns:
        Validity rate as float in [0, 1]

    Raises:
        ValueError: If inputs are invalid
    """
    if counterfactuals is None or target_classes is None or classifier is None:
        raise ValueError("counterfactuals, target_classes, and classifier are required")

    if len(counterfactuals) == 0:
        raise ValueError("counterfactuals tensor is empty")

    if len(counterfactuals) != len(target_classes):
        raise ValueError(
            f"Mismatch: {len(counterfactuals)} counterfactuals vs {len(target_classes)} target classes"
        )

    classifier.eval()
    device = next(classifier.parameters()).device
    counterfactuals = counterfactuals.to(device)
    target_classes = target_classes.to(device)

    with torch.no_grad():
        logits = classifier(counterfactuals)
        predictions = torch.argmax(logits, dim=1)
        valid = (predictions == target_classes).float()
        validity_rate = valid.mean().item()

    return validity_rate


# ==================== 2. PREDICTION GAIN ====================

def compute_prediction_gain(
    originals: torch.Tensor,
    counterfactuals: torch.Tensor,
    target_classes: torch.Tensor,
    classifier: nn.Module
) -> float:
    """
    Compute Prediction Gain: average confidence increase for target class.

    ΔP = (1/N) Σ [P(y'|x_cf) - P(y'|x)]

    Args:
        originals: Original instances [N, *input_shape]
        counterfactuals: Generated counterfactuals [N, *input_shape]
        target_classes: Target class labels [N]
        classifier: Trained classifier model

    Returns:
        Mean prediction gain as float

    Raises:
        ValueError: If inputs are invalid
    """
    if originals is None or counterfactuals is None or target_classes is None or classifier is None:
        raise ValueError("All arguments are required: originals, counterfactuals, target_classes, classifier")

    if len(originals) != len(counterfactuals) or len(originals) != len(target_classes):
        raise ValueError(
            f"Size mismatch: {len(originals)} originals, {len(counterfactuals)} CFs, {len(target_classes)} targets"
        )

    classifier.eval()
    device = next(classifier.parameters()).device
    originals = originals.to(device)
    counterfactuals = counterfactuals.to(device)
    target_classes = target_classes.to(device)

    with torch.no_grad():
        # Get probabilities for target class
        probs_orig = F.softmax(classifier(originals), dim=1)
        probs_cf = F.softmax(classifier(counterfactuals), dim=1)

        # Extract target class probabilities
        target_probs_orig = probs_orig.gather(1, target_classes.unsqueeze(1)).squeeze()
        target_probs_cf = probs_cf.gather(1, target_classes.unsqueeze(1)).squeeze()

        # Compute gain
        gain = target_probs_cf - target_probs_orig
        mean_gain = gain.mean().item()

    return mean_gain


# ==================== 3. PROXIMITY (L1) ====================

def compute_proximity_l1(
    originals: torch.Tensor,
    counterfactuals: torch.Tensor
) -> float:
    """
    Compute Proximity: average L1 distance between originals and counterfactuals.

    Proximity = (1/N) Σ ||x - x_cf||₁

    Args:
        originals: Original instances [N, *input_shape]
        counterfactuals: Generated counterfactuals [N, *input_shape]

    Returns:
        Mean L1 distance as float

    Raises:
        ValueError: If inputs are invalid
    """
    if originals is None or counterfactuals is None:
        raise ValueError("Both originals and counterfactuals are required")

    if originals.shape != counterfactuals.shape:
        raise ValueError(
            f"Shape mismatch: originals {originals.shape} vs counterfactuals {counterfactuals.shape}"
        )

    if len(originals) == 0:
        raise ValueError("Empty input tensors")

    # Flatten to [N, D] and compute L1 distance
    orig_flat = originals.reshape(len(originals), -1)
    cf_flat = counterfactuals.reshape(len(counterfactuals), -1)

    l1_distances = torch.abs(orig_flat - cf_flat).sum(dim=1)
    mean_l1 = l1_distances.mean().item()

    return mean_l1


# ==================== 6. INVALIDATION RATE ====================

def compute_invalidation_rate(
    counterfactuals: torch.Tensor,
    target_classes: torch.Tensor,
    classifier: nn.Module,
    sigma: float = 0.05,
    n_samples: int = 1,
    seed: Optional[int] = 42
) -> float:
    """
    Compute Invalidation Rate: percentage of counterfactuals that flip when noise is added.

    IR_σ = 1 - (1/N) Σ 1[f(x_cf + ε) = y'],  ε ~ N(0, σ²I)

    Args:
        counterfactuals: Generated counterfactuals [N, *input_shape]
        target_classes: Target class labels [N]
        classifier: Trained classifier model
        sigma: Gaussian noise standard deviation (default: 0.05)
        n_samples: Number of noise samples per counterfactual (default: 1)
        seed: Random seed for reproducibility

    Returns:
        Invalidation rate as float in [0, 1]

    Raises:
        ValueError: If inputs are invalid
    """
    if counterfactuals is None or target_classes is None or classifier is None:
        raise ValueError("counterfactuals, target_classes, and classifier are required")

    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    classifier.eval()
    device = next(classifier.parameters()).device
    counterfactuals = counterfactuals.to(device)
    target_classes = target_classes.to(device)

    N = len(counterfactuals)
    original_shape = counterfactuals.shape

    # Generate noisy samples: [n_samples, N, *input_shape]
    noise = torch.randn(n_samples, *original_shape, device=device) * sigma
    noisy_cfs = counterfactuals.unsqueeze(0) + noise  # [n_samples, N, *input_shape]

    # Reshape to [n_samples * N, *input_shape]
    noisy_cfs_flat = noisy_cfs.reshape(-1, *original_shape[1:])

    # Expand target classes
    expanded_targets = target_classes.repeat(n_samples)

    # Evaluate validity of noisy counterfactuals
    with torch.no_grad():
        logits = classifier(noisy_cfs_flat)
        predictions = torch.argmax(logits, dim=1)
        valid = (predictions == expanded_targets).float()

        # Average over noise samples
        valid_reshaped = valid.reshape(n_samples, N).mean(dim=0)  # [N]
        validity_rate = valid_reshaped.mean().item()

    invalidation_rate = 1.0 - validity_rate

    return invalidation_rate


# ==================== 7. ROBUSTNESS ====================

def compute_robustness(
    counterfactuals: torch.Tensor,
    target_classes: torch.Tensor,
    classifier: nn.Module,
    sigma: float = 0.05,
    n_samples: int = 1,
    seed: Optional[int] = 42
) -> float:
    """
    Compute Robustness: percentage of counterfactuals that remain valid under noise.

    Robustness_σ = 1 - IR_σ

    Args:
        counterfactuals: Generated counterfactuals [N, *input_shape]
        target_classes: Target class labels [N]
        classifier: Trained classifier model
        sigma: Gaussian noise standard deviation (default: 0.05)
        n_samples: Number of noise samples per counterfactual (default: 1)
        seed: Random seed for reproducibility

    Returns:
        Robustness as float in [0, 1]

    Raises:
        ValueError: If inputs are invalid
    """
    ir = compute_invalidation_rate(
        counterfactuals, target_classes, classifier, sigma, n_samples, seed
    )
    return 1.0 - ir


# ==================== 8. LATENCY ====================

def compute_latency(
    dataset: torch.utils.data.Dataset,
    target_classes: torch.Tensor,
    cf_generator: Callable,
    num_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute Latency: average time to generate one counterfactual.

    Latency = (1/N) Σ time(CF(x_i, y'_i))

    Args:
        dataset: Dataset containing original samples
        target_classes: Target class labels [N]
        cf_generator: Function that generates counterfactuals: (x, y') -> x_cf
        num_samples: Number of samples to time (default: all)
        device: Device for computation

    Returns:
        Mean latency in seconds per counterfactual

    Raises:
        ValueError: If inputs are invalid
    """
    if dataset is None or target_classes is None or cf_generator is None:
        raise ValueError("dataset, target_classes, and cf_generator are required")

    if len(dataset) == 0:
        raise ValueError("Empty dataset")

    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    if num_samples > len(target_classes):
        raise ValueError(
            f"num_samples ({num_samples}) exceeds number of target classes ({len(target_classes)})"
        )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    times = []

    for i in range(num_samples):
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(device)  # [1, *input_shape]
        y_target = target_classes[i].unsqueeze(0).to(device)  # [1]

        # Time the generation
        start = time.time()
        with torch.no_grad():
            _ = cf_generator(x, y_target)

        # Synchronize GPU if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.time()
        times.append(end - start)

    mean_latency = float(np.mean(times))

    return mean_latency


# ==================== COMPREHENSIVE EVALUATION ====================

def evaluate_all_metrics(
    originals: torch.Tensor,
    counterfactuals: torch.Tensor,
    target_classes: torch.Tensor,
    classifier: nn.Module,
    dataset: Optional[torch.utils.data.Dataset] = None,
    cf_generator: Optional[Callable] = None,
    global_autoencoder: Optional[nn.Module] = None,
    class_autoencoders: Optional[Dict[int, nn.Module]] = None,
    likelihood: str = 'bernoulli',
    sigma: float = 0.05,
    n_samples_ir: int = 1,
    num_samples_latency: Optional[int] = None,
    seed: Optional[int] = 42
) -> Dict[str, float]:
    """
    Compute all 8 evaluation metrics.

    Args:
        originals: Original instances [N, *input_shape]
        counterfactuals: Generated counterfactuals [N, *input_shape]
        target_classes: Target class labels [N]
        classifier: Trained classifier model
        dataset: Dataset for latency computation (optional)
        cf_generator: CF generation function for latency (optional)
        global_autoencoder: Global autoencoder for Plausibility (AE) (optional)
        class_autoencoders: Per-class autoencoders for Plausibility (tc-AE) (optional)
        likelihood: Likelihood type for plausibility ('bernoulli' or 'gaussian')
        sigma: Noise level for IR/Robustness (default: 0.05)
        n_samples_ir: Number of noise samples for IR (default: 1)
        num_samples_latency: Number of samples for latency computation
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing all metrics:
        {
            'validity_rate': float,
            'prediction_gain': float,
            'proximity_l1': float,
            'plausibility_ae': float (if global_autoencoder provided),
            'plausibility_tc_ae': float (if class_autoencoders provided),
            'invalidation_rate': float,
            'robustness': float,
            'latency_seconds': float (if dataset and cf_generator provided)
        }

    Raises:
        ValueError: If required inputs are invalid
    """
    results = {}

    # 1. Validity Rate
    try:
        validity = compute_validity_rate(counterfactuals, target_classes, classifier)
        results['validity_rate'] = validity
        logger.info(f"Validity Rate: {validity:.4f}")
    except Exception as e:
        logger.error(f"Error computing validity rate: {e}")
        raise

    # 2. Prediction Gain
    try:
        gain = compute_prediction_gain(originals, counterfactuals, target_classes, classifier)
        results['prediction_gain'] = gain
        logger.info(f"Prediction Gain: {gain:.4f}")
    except Exception as e:
        logger.error(f"Error computing prediction gain: {e}")
        raise

    # 3. Proximity (L1)
    try:
        proximity = compute_proximity_l1(originals, counterfactuals)
        results['proximity_l1'] = proximity
        logger.info(f"Proximity (L1): {proximity:.4f}")
    except Exception as e:
        logger.error(f"Error computing proximity: {e}")
        raise

    # 4. Plausibility (AE) - optional
    if global_autoencoder is not None:
        try:
            from .plausibility_ae import compute_plausibility_ae
            plaus_ae = compute_plausibility_ae(counterfactuals, global_autoencoder, likelihood)
            results['plausibility_ae'] = plaus_ae
            logger.info(f"Plausibility (AE): {plaus_ae:.4f}")
        except Exception as e:
            logger.warning(f"Error computing plausibility (AE): {e}")

    # 5. Plausibility (tc-AE) - optional
    if class_autoencoders is not None:
        try:
            from .plausibility_ae import compute_plausibility_tc_ae
            plaus_tc = compute_plausibility_tc_ae(
                counterfactuals, target_classes, class_autoencoders, likelihood
            )
            results['plausibility_tc_ae'] = plaus_tc
            logger.info(f"Plausibility (tc-AE): {plaus_tc:.4f}")
        except Exception as e:
            logger.warning(f"Error computing plausibility (tc-AE): {e}")

    # 6. Invalidation Rate
    try:
        ir = compute_invalidation_rate(
            counterfactuals, target_classes, classifier, sigma, n_samples_ir, seed
        )
        results['invalidation_rate'] = ir
        logger.info(f"Invalidation Rate (σ={sigma}): {ir:.4f}")
    except Exception as e:
        logger.error(f"Error computing invalidation rate: {e}")
        raise

    # 7. Robustness
    try:
        robustness = 1.0 - ir  # Already computed above
        results['robustness'] = robustness
        logger.info(f"Robustness (σ={sigma}): {robustness:.4f}")
    except Exception as e:
        logger.error(f"Error computing robustness: {e}")
        raise

    # 8. Latency - optional
    if dataset is not None and cf_generator is not None:
        try:
            latency = compute_latency(
                dataset, target_classes, cf_generator, num_samples_latency
            )
            results['latency_seconds'] = latency
            logger.info(f"Latency: {latency:.6f} seconds")
        except Exception as e:
            logger.warning(f"Error computing latency: {e}")

    logger.info("All metrics computed successfully")
    return results
