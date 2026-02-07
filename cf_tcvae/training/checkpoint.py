"""Checkpoint saving and loading utilities for CF-TCVAE training.

Handles model checkpoints, optimizer states, learning rate scheduler states,
and training metadata. Follows the NO FALLBACKS policy.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and loading for CF-TCVAE training.

    Features:
    - Save/load model state dict
    - Save/load optimizer and scheduler states
    - Track training metadata (epoch, best metrics)
    - Automatic backup of best checkpoints
    - Validation of checkpoint integrity

    Attributes:
        checkpoint_dir: Directory where checkpoints are saved
        save_best_only: If True, only save checkpoint when validation improves
        best_metric_name: Name of metric to track for best checkpoint
        best_metric_mode: 'min' or 'max' for best metric tracking
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        best_metric_name: str = 'val_loss',
        best_metric_mode: str = 'min'
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints (will be created if needed)
            save_best_only: Whether to only save when validation metric improves
            best_metric_name: Name of metric to track (e.g., 'val_loss')
            best_metric_mode: 'min' (lower is better) or 'max' (higher is better)

        Raises:
            ValueError: If best_metric_mode is not 'min' or 'max'
        """
        if best_metric_mode not in ['min', 'max']:
            raise ValueError(f"best_metric_mode must be 'min' or 'max', got '{best_metric_mode}'")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode

        # Track best metric value
        if best_metric_mode == 'min':
            self.best_metric_value = float('inf')
        else:
            self.best_metric_value = float('-inf')

        self.best_epoch = None

        logger.info(f"CheckpointManager initialized: dir={checkpoint_dir}, "
                    f"best_metric={best_metric_name} ({best_metric_mode})")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: CF-TCVAE model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler (or None)
            epoch: Current epoch number
            metrics: Dictionary of metrics (must include best_metric_name if save_best_only)
            is_best: Whether this is the best checkpoint so far
            extra_state: Additional state to save (e.g., TC estimator memory banks)

        Returns:
            Path to saved checkpoint file

        Raises:
            ValueError: If save_best_only=True but best_metric_name not in metrics
        """
        # Validate metrics if save_best_only
        if self.save_best_only and self.best_metric_name not in metrics:
            raise ValueError(
                f"save_best_only=True but best_metric_name '{self.best_metric_name}' "
                f"not found in metrics. Available: {list(metrics.keys())}"
            )

        # Create checkpoint state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric_name': self.best_metric_name,
            'best_metric_value': self.best_metric_value,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint['extra_state'] = extra_state

        # Determine checkpoint filename
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            logger.info(f"Saving best checkpoint at epoch {epoch}: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            logger.info(f"Saving checkpoint at epoch {epoch}: {checkpoint_path}")

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path)

    def should_save_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """Determine if checkpoint should be saved based on metrics.

        Args:
            metrics: Dictionary of current metrics

        Returns:
            True if checkpoint should be saved

        Raises:
            ValueError: If best_metric_name not in metrics when save_best_only=True
        """
        if not self.save_best_only:
            return True

        if self.best_metric_name not in metrics:
            raise ValueError(
                f"best_metric_name '{self.best_metric_name}' not found in metrics. "
                f"Available: {list(metrics.keys())}"
            )

        current_value = metrics[self.best_metric_name]

        # Check if metric improved
        if self.best_metric_mode == 'min':
            improved = current_value < self.best_metric_value
        else:
            improved = current_value > self.best_metric_value

        if improved:
            self.best_metric_value = current_value
            return True

        return False

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to ('cuda' or 'cpu')
            strict: Whether to strictly enforce state dict key matching

        Returns:
            Dictionary containing checkpoint metadata (epoch, metrics, extra_state)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info(f"Loaded model state from epoch {checkpoint['epoch']}")

            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state")

            # Load scheduler state if provided
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state")

            # Extract metadata
            metadata = {
                'epoch': checkpoint['epoch'],
                'metrics': checkpoint.get('metrics', {}),
                'best_metric_name': checkpoint.get('best_metric_name'),
                'best_metric_value': checkpoint.get('best_metric_value'),
                'extra_state': checkpoint.get('extra_state'),
            }

            logger.info(f"Checkpoint loaded successfully: epoch={metadata['epoch']}, "
                        f"metrics={metadata['metrics']}")

            return metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint if it exists.

        Returns:
            Path to best checkpoint, or None if not found
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return str(best_path)
        return None

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to most recent epoch checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        # Find all epoch checkpoints
        epoch_checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if not epoch_checkpoints:
            return None

        # Sort by epoch number (extract from filename)
        def get_epoch_num(path):
            filename = path.stem  # e.g., 'checkpoint_epoch_42'
            return int(filename.split('_')[-1])

        latest = max(epoch_checkpoints, key=get_epoch_num)
        return str(latest)

    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Remove old epoch checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep

        Raises:
            ValueError: If keep_last_n < 1
        """
        if keep_last_n < 1:
            raise ValueError(f"keep_last_n must be >= 1, got {keep_last_n}")

        # Find all epoch checkpoints (excluding best_model.pt)
        epoch_checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if len(epoch_checkpoints) <= keep_last_n:
            return

        # Sort by epoch number
        def get_epoch_num(path):
            filename = path.stem
            return int(filename.split('_')[-1])

        epoch_checkpoints.sort(key=get_epoch_num)

        # Remove oldest checkpoints
        to_remove = epoch_checkpoints[:-keep_last_n]
        for checkpoint_path in to_remove:
            checkpoint_path.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint_path}")


def save_model_only(
    model: torch.nn.Module,
    save_path: str,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None
):
    """Save model state dict only (no optimizer/scheduler).

    Useful for saving final trained models for evaluation/deployment.

    Args:
        model: Model to save
        save_path: Path to save model (will be created)
        epoch: Optional epoch number to include
        metrics: Optional metrics dictionary to include

    Raises:
        RuntimeError: If saving fails
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    try:
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model to: {save_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {save_path}: {str(e)}")


def load_model_only(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> Dict[str, Any]:
    """Load model state dict only (no optimizer/scheduler).

    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load to ('cuda' or 'cpu')
        strict: Whether to strictly enforce state dict key matching

    Returns:
        Dictionary with 'epoch' and 'metrics' if available

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        metadata = {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
        }

        logger.info(f"Loaded model from: {checkpoint_path}")
        return metadata

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {str(e)}")
