"""
Example: Walk-Forward Training

This example demonstrates walk-forward optimization for time-series model training,
ensuring no lookahead bias and proper out-of-sample evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training.walk_forward import WalkForwardOptimizer, WindowType, walk_forward_validation
from quantml.training.metrics import sharpe_ratio, information_coefficient
from quantml.training.losses import mse_loss


def generate_returns_data(n_samples=500):
    """Generate synthetic returns data."""
    import random
    returns = []
    for _ in range(n_samples):
        ret = random.gauss(0.001, 0.02)  # Mean 0.1% daily, 2% std
        returns.append(ret)
    return returns


def create_features_from_returns(returns, lookback=10):
    """Create features from returns (lagged returns)."""
    features = []
    for i in range(lookback, len(returns)):
        feature_vec = returns[i-lookback:i]
        features.append(feature_vec)
    return features, returns[lookback:]


def train_model(model, X_train, y_train):
    """Train model on data."""
    from quantml.optim import Adam
    from quantml.training import QuantTrainer
    
    optimizer = Adam(model.parameters(), lr=0.01)
    trainer = QuantTrainer(model, optimizer, loss_fn=mse_loss)
    
    # Quick training (few epochs for example)
    trainer.train(X_train, y_train, epochs=20, batch_size=32, verbose=False)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    predictions = []
    for x in X_test:
        x_tensor = Tensor([x])
        pred = model.forward(x_tensor)
        pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
        predictions.append(pred_val)
    
    # Calculate metrics
    ic = information_coefficient(predictions, y_test)
    
    # Calculate returns from predictions (simplified)
    returns = [(p - y) ** 2 for p, y in zip(predictions, y_test)]  # MSE as proxy
    sharpe = sharpe_ratio(returns, annualize=False)
    
    return {
        'ic': ic,
        'sharpe': sharpe,
        'mse': sum((p - y) ** 2 for p, y in zip(predictions, y_test)) / len(predictions)
    }


def main():
    """Run walk-forward training example."""
    print("=== Walk-Forward Training Example ===\n")
    
    # Generate data
    print("Generating synthetic returns data...")
    returns = generate_returns_data(n_samples=500)
    
    # Create features
    features, targets = create_features_from_returns(returns, lookback=10)
    print(f"Created {len(features)} samples with {len(features[0])} features\n")
    
    # Create walk-forward optimizer
    wfo = WalkForwardOptimizer(
        window_type=WindowType.EXPANDING,
        train_size=100,  # Initial training window
        test_size=20,    # Test window
        step_size=20     # Step forward by test_size
    )
    
    print("Performing walk-forward validation...")
    splits = wfo.get_splits(len(features), n_splits=5)
    print(f"Created {len(splits)} train/test splits\n")
    
    # Walk-forward validation
    results = []
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Split {split_idx + 1}/{len(splits)}:")
        print(f"  Train: indices {train_idx[0]}-{train_idx[-1]} ({len(train_idx)} samples)")
        print(f"  Test:  indices {test_idx[0]}-{test_idx[-1]} ({len(test_idx)} samples)")
        
        # Get data
        X_train = [features[i] for i in train_idx]
        y_train = [targets[i] for i in train_idx]
        X_test = [features[i] for i in test_idx]
        y_test = [targets[i] for i in test_idx]
        
        # Create and train model
        model = Linear(in_features=len(features[0]), out_features=1, bias=True)
        trained_model = train_model(model, X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(trained_model, X_test, y_test)
        results.append(metrics)
        
        print(f"  IC: {metrics['ic']:.4f}, Sharpe: {metrics['sharpe']:.4f}, MSE: {metrics['mse']:.6f}\n")
    
    # Aggregate results
    avg_ic = sum(r['ic'] for r in results) / len(results)
    avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
    avg_mse = sum(r['mse'] for r in results) / len(results)
    
    print("=== Aggregate Results ===")
    print(f"Average IC: {avg_ic:.4f}")
    print(f"Average Sharpe: {avg_sharpe:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print("\nExample completed!")


if __name__ == "__main__":
    main()

