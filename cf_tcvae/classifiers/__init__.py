"""Pre-trained black-box classifiers for MNIST and Adult datasets.

This module provides pre-trained CNN classifiers for MNIST digit classification
and MLP classifiers for Adult Income binary classification.

The classifiers are used as frozen black-box models for generating counterfactual
explanations with CF-TCVAE.

Example usage:
    >>> from cf_tcvae.classifiers import load_classifier
    >>>
    >>> # Load MNIST classifier
    >>> mnist_clf, _ = load_classifier('mnist', freeze=True)
    >>>
    >>> # Load Adult classifier
    >>> adult_clf, _ = load_classifier('adult', freeze=True)
"""

from .mnist_cnn import (
    MNISTCNNBasic,
    MNISTCNNAdvanced,
    MNISTCNNEfficient,
    create_mnist_classifier
)

from .adult_mlp import (
    AdultMLPBasic,
    AdultMLPAdvanced,
    AdultMLPEfficient,
    create_adult_classifier
)

from .utils import (
    load_mnist_classifier,
    load_adult_classifier,
    load_classifier,
    verify_frozen,
    count_parameters,
    get_model_info
)

__all__ = [
    # MNIST models
    'MNISTCNNBasic',
    'MNISTCNNAdvanced',
    'MNISTCNNEfficient',
    'create_mnist_classifier',

    # Adult models
    'AdultMLPBasic',
    'AdultMLPAdvanced',
    'AdultMLPEfficient',
    'create_adult_classifier',

    # Loading utilities
    'load_mnist_classifier',
    'load_adult_classifier',
    'load_classifier',
    'verify_frozen',
    'count_parameters',
    'get_model_info',
]
