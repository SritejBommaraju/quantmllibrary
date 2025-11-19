"""
Training utilities for quant models.

This module provides QuantTrainer class with training loop, early stopping,
checkpointing, and metrics tracking.
"""

from typing import Optional, Callable, Dict, Any, List
from quantml.tensor import Tensor
from quantml import ops
from quantml.training.gradient_clipping import GradientNormClipper, GradientValueClipper, AdaptiveClipper

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class QuantTrainer:
    """
    Trainer class for quant model training.
    
    Provides training loop with early stopping, checkpointing, and metrics tracking.
    
    Attributes:
        model: Model to train
        optimizer: Optimizer
        loss_fn: Loss function
        metrics: List of metric functions to track
    
    Examples:
        >>> trainer = QuantTrainer(model, optimizer, loss_fn=mse_loss)
        >>> trainer.train(X_train, y_train, X_val, y_val, epochs=100)
    """
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Callable,
        metrics: Optional[List[Callable]] = None,
        gradient_clipper: Optional[Any] = None,
        accumulation_steps: int = 1
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer (SGD, Adam, etc.)
            loss_fn: Loss function
            metrics: Optional list of metric functions
            gradient_clipper: Optional gradient clipper (GradientNormClipper, etc.)
            accumulation_steps: Number of steps to accumulate gradients before optimizer step
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics else []
        self.gradient_clipper = gradient_clipper
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_step(self, x: Tensor, y: Tensor) -> float:
        """
        Perform a single training step.
        
        Args:
            x: Input features
            y: Targets
        
        Returns:
            Loss value
        """
        # Forward pass
        pred = self.model.forward(x)
        
        # Compute loss (scale by accumulation steps for correct averaging)
        loss = self.loss_fn(pred, y)
        if self.accumulation_steps > 1:
            loss = ops.mul(loss, 1.0 / self.accumulation_steps)
        
        # Backward pass
        if loss.requires_grad:
            loss.backward()
        
        self.accumulation_counter += 1
        
        # Apply gradient clipping if enabled
        if self.gradient_clipper is not None and self.accumulation_counter % self.accumulation_steps == 0:
            params = self.model.parameters() if hasattr(self.model, 'parameters') else []
            self.gradient_clipper(params)
        
        # Optimizer step only after accumulation
        if self.accumulation_counter % self.accumulation_steps == 0:
            self.optimizer.step()
            self.model.zero_grad()
            self.accumulation_counter = 0
        
        # Get loss value (unscale for reporting)
        loss_value = self._get_value(loss)
        if self.accumulation_steps > 1:
            loss_value = loss_value * self.accumulation_steps
        return loss_value
    
    def validate(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """
        Validate model on data.
        
        Args:
            x: Input features
            y: Targets
        
        Returns:
            Dictionary of metrics
        """
        # Forward pass
        pred = self.model.forward(x)
        
        # Compute loss
        loss = self.loss_fn(pred, y)
        metrics_dict = {'loss': self._get_value(loss)}
        
        # Compute additional metrics
        for metric_fn in self.metrics:
            try:
                metric_val = metric_fn(pred, y)
                if isinstance(metric_val, Tensor):
                    metric_val = self._get_value(metric_val)
                metrics_dict[metric_fn.__name__] = metric_val
            except Exception:
                pass
        
        return metrics_dict
    
    def train(
        self,
        X_train: List,
        y_train: List,
        X_val: Optional[List] = None,
        y_val: Optional[List] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Early stopping config (patience, min_delta)
            verbose: Whether to print progress
        
        Returns:
            Training history
        """
        from quantml.training.data_loader import QuantDataLoader
        
        train_loader = QuantDataLoader(X_train, y_train, batch_size=batch_size, shuffle=False)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = QuantDataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = early_stopping.get('patience', 10) if early_stopping else None
        min_delta = early_stopping.get('min_delta', 0.0) if early_stopping else 0.0
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(loss)
            
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self._validate_loader(val_loader)
                val_loss = val_metrics.get('loss', 0.0)
                self.history['val_loss'].append(val_loss)
                
                # Early stopping check
                if patience is not None:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.6f}", end="")
                if val_loader is not None:
                    print(f", Val Loss = {val_loss:.6f}")
                else:
                    print()
        
        return self.history
    
    def _validate_loader(self, loader) -> Dict[str, float]:
        """Validate on data loader."""
        all_metrics = []
        for x_batch, y_batch in loader:
            metrics = self.validate(x_batch, y_batch)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_metrics
    
    def _get_value(self, tensor: Tensor) -> float:
        """Extract scalar value from tensor."""
        if isinstance(tensor.data, list):
            if isinstance(tensor.data[0], list):
                return float(tensor.data[0][0])
            return float(tensor.data[0])
        return float(tensor.data)

