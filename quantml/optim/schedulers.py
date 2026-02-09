"""
Learning rate schedulers for QuantML.

Provides various learning rate scheduling strategies for training optimization.
"""

from typing import List, Optional, Any
from abc import ABC, abstractmethod
import math


class LRScheduler(ABC):
    """
    Base class for learning rate schedulers.
    
    All schedulers should inherit from this class and implement
    the get_lr() and step() methods.
    """
    
    def __init__(self, optimizer: Any, last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            last_epoch: The index of the last epoch
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group.get('lr', optimizer.lr) if isinstance(group, dict) else optimizer.lr 
                        for group in getattr(optimizer, 'param_groups', [optimizer])]
    
    @abstractmethod
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        pass
    
    def step(self, epoch: Optional[int] = None):
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch (if None, uses last_epoch + 1)
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        lrs = self.get_lr()
        # Update optimizer learning rates
        if hasattr(self.optimizer, 'param_groups'):
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        else:
            self.optimizer.lr = lrs[0] if lrs else self.optimizer.lr


class StepLR(LRScheduler):
    """Step learning rate scheduler - decays LR by gamma every step_size epochs."""
    
    def __init__(self, optimizer: Any, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        """
        Initialize StepLR scheduler.
        
        Args:
            optimizer: The optimizer
            step_size: Period of learning rate decay
            gamma: Multiplicative factor for decay
            last_epoch: The index of the last epoch
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate."""
        return [base_lr * (self.gamma ** (self.last_epoch // self.step_size)) 
                for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer: Any, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        """
        Initialize CosineAnnealingLR scheduler.
        
        Args:
            optimizer: The optimizer
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            last_epoch: The index of the last epoch
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate."""
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class WarmupLR(LRScheduler):
    """Warmup learning rate scheduler."""
    
    def __init__(self, optimizer: Any, warmup_steps: int, warmup_type: str = 'linear', last_epoch: int = -1):
        """
        Initialize WarmupLR scheduler.
        
        Args:
            optimizer: The optimizer
            warmup_steps: Number of warmup steps
            warmup_type: Type of warmup ('linear' or 'cosine')
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.warmup_type = warmup_type
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate."""
        if self.last_epoch < self.warmup_steps:
            if self.warmup_type == 'linear':
                factor = (self.last_epoch + 1) / self.warmup_steps
            else:  # cosine
                factor = (1 + math.cos(math.pi * (1 - (self.last_epoch + 1) / self.warmup_steps))) / 2
            return [base_lr * factor for base_lr in self.base_lrs]
        return self.base_lrs


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""
    
    def __init__(self, optimizer: Any, mode: str = 'min', factor: float = 0.1, 
                 patience: int = 10, threshold: float = 1e-4, min_lr: float = 0.0):
        """
        Initialize ReduceLROnPlateau scheduler.
        
        Args:
            optimizer: The optimizer
            mode: 'min' or 'max' - whether to reduce LR when metric stops decreasing/increasing
            factor: Factor to multiply LR by
            patience: Number of epochs with no improvement before reducing LR
            threshold: Threshold for measuring improvement
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        self.base_lrs = [optimizer.lr]
    
    def step(self, metrics: float):
        """
        Step the scheduler based on metrics.
        
        Args:
            metrics: Current metric value
        """
        if self.best is None:
            self.best = metrics
        else:
            if self.mode == 'min':
                is_better = metrics < self.best - self.threshold
            else:
                is_better = metrics > self.best + self.threshold
            
            if is_better:
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            
            if self.num_bad_epochs >= self.patience:
                self._reduce_lr()
                self.num_bad_epochs = 0
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
        self.optimizer.lr = new_lr


class CyclicLR(LRScheduler):
    """Cyclic learning rate scheduler."""
    
    def __init__(self, optimizer: Any, base_lr: float, max_lr: float, 
                 step_size_up: int = 2000, step_size_down: Optional[int] = None,
                 mode: str = 'triangular', gamma: float = 1.0, last_epoch: int = -1):
        """
        Initialize CyclicLR scheduler.
        
        Args:
            optimizer: The optimizer
            base_lr: Lower bound of learning rate
            max_lr: Upper bound of learning rate
            step_size_up: Number of steps to increase LR
            step_size_down: Number of steps to decrease LR (if None, equals step_size_up)
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: Scaling factor for 'exp_range' mode
            last_epoch: The index of the last epoch
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.mode = mode
        self.gamma = gamma
        self.step_size = self.step_size_up + self.step_size_down
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate."""
        cycle = math.floor(1 + self.last_epoch / self.step_size)
        # x is the fractional position within the current cycle (0 to 1)
        x = self.last_epoch / self.step_size - (cycle - 1)

        # Compute scale factor (0 to 1 triangular wave)
        up_fraction = self.step_size_up / self.step_size
        if x <= up_fraction:
            # Ascending phase: scale goes from 0 to 1
            scale = x / up_fraction if up_fraction > 0 else 1.0
        else:
            # Descending phase: scale goes from 1 to 0
            down_fraction = self.step_size_down / self.step_size
            scale = (1.0 - x) / down_fraction if down_fraction > 0 else 0.0

        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, scale)
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, scale) / (2 ** (cycle - 1))
        else:  # exp_range
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, scale) * (self.gamma ** self.last_epoch)

        return [lr for _ in self.base_lrs]


class OneCycleLR(LRScheduler):
    """One cycle learning rate scheduler."""
    
    def __init__(self, optimizer: Any, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos',
                 div_factor: float = 25.0, final_div_factor: float = 10000.0, last_epoch: int = -1):
        """
        Initialize OneCycleLR scheduler.
        
        Args:
            optimizer: The optimizer
            max_lr: Maximum learning rate
            total_steps: Total number of steps
            pct_start: Percentage of steps for warmup
            anneal_strategy: 'cos' or 'linear' annealing
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = initial_lr / final_div_factor
            last_epoch: The index of the last epoch
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate."""
        if self.last_epoch < self.total_steps * self.pct_start:
            # Warmup phase
            pct = self.last_epoch / (self.total_steps * self.pct_start)
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (1 + math.cos(math.pi * (1 - pct))) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # Annealing phase
            pct = (self.last_epoch - self.total_steps * self.pct_start) / (self.total_steps * (1 - self.pct_start))
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * pct)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * pct
        
        return [lr for _ in self.base_lrs]

