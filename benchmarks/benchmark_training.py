"""
Benchmark training performance.

Measures training speed with various optimizers and configurations.
"""

import time
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import SGD, Adam, RMSProp
from quantml.training import QuantTrainer
from quantml.training.losses import mse_loss


def benchmark_training(optimizer_class, optimizer_kwargs, n_epochs=10, batch_size=32):
    """Benchmark training with a specific optimizer."""
    print(f"\nBenchmarking {optimizer_class.__name__}...")
    
    # Create model and data
    model = Linear(in_features=50, out_features=1, bias=True)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    trainer = QuantTrainer(model, optimizer, mse_loss)
    
    # Generate synthetic data
    X = [Tensor([[1.0] * 50]) for _ in range(batch_size)]
    y = [Tensor([[1.0]]) for _ in range(batch_size)]
    
    # Warmup
    for _ in range(2):
        for x, target in zip(X, y):
            trainer.train_step(x, target)
    
    # Benchmark
    start = time.time()
    for epoch in range(n_epochs):
        for x, target in zip(X, y):
            trainer.train_step(x, target)
    elapsed = time.time() - start
    
    total_steps = n_epochs * batch_size
    print(f"Time: {elapsed:.4f}s ({elapsed/total_steps*1000:.4f}ms per step)")
    return elapsed


if __name__ == '__main__':
    print("=" * 60)
    print("QuantML Training Performance Benchmarks")
    print("=" * 60)
    
    benchmark_training(SGD, {'lr': 0.01}, n_epochs=5)
    benchmark_training(Adam, {'lr': 0.001}, n_epochs=5)
    benchmark_training(RMSProp, {'lr': 0.01}, n_epochs=5)
    
    print("\n" + "=" * 60)
    print("Training benchmarks completed!")
    print("=" * 60)

