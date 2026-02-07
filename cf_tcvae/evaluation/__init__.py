"""
Evaluation module for CF-TCVAE counterfactual explanations.

Provides 8 metrics from Table 1 of the paper:
1. Validity Rate (%)
2. Prediction Gain
3. Proximity (L1)
4. Plausibility (AE)
5. Plausibility (tc-AE)
6. Invalidation Rate
7. Robustness
8. Latency (s)
"""

from .metrics import (
    compute_validity_rate,
    compute_prediction_gain,
    compute_proximity_l1,
    compute_invalidation_rate,
    compute_robustness,
    compute_latency,
    evaluate_all_metrics,
)

from .plausibility_ae import (
    MNISTAutoencoder,
    TabularAutoencoder,
    train_autoencoder,
    train_global_autoencoder,
    train_class_autoencoders,
    compute_plausibility_ae,
    compute_plausibility_tc_ae,
    save_autoencoder,
    load_autoencoder,
)

__all__ = [
    # Individual metrics
    'compute_validity_rate',
    'compute_prediction_gain',
    'compute_proximity_l1',
    'compute_invalidation_rate',
    'compute_robustness',
    'compute_latency',

    # Comprehensive evaluation
    'evaluate_all_metrics',

    # Autoencoder architectures
    'MNISTAutoencoder',
    'TabularAutoencoder',

    # Autoencoder training
    'train_autoencoder',
    'train_global_autoencoder',
    'train_class_autoencoders',

    # Plausibility metrics
    'compute_plausibility_ae',
    'compute_plausibility_tc_ae',

    # Autoencoder utilities
    'save_autoencoder',
    'load_autoencoder',
]
