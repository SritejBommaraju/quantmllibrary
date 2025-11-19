"""
Example: Futures Model Pipeline

This example demonstrates a complete pipeline for a futures trading model:
1. Load/stream market data
2. Apply quant-specific operations (EMA, volatility, etc.)
3. Train a small model
4. Perform online inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml import ops
from quantml import time_series
from quantml.streaming import StreamingTensor


def generate_futures_data(n_bars=500):
    """Generate synthetic futures bar data."""
    prices = []
    volumes = []
    base_price = 5000.0
    
    for i in range(n_bars):
        # Simulate price movement
        change = (i % 50 - 25) * 0.5
        trend = i * 0.1
        noise = (i % 7 - 3) * 0.2
        price = base_price + change + trend + noise
        prices.append(price)
        volumes.append(100.0 + (i % 10) * 5.0)
        base_price = price
    
    return prices, volumes


def create_features(prices, volumes):
    """Create features using quant operations."""
    # Convert to tensors
    price_tensor = Tensor([prices])
    volume_tensor = Tensor([volumes])
    
    # Compute features
    ema_20 = time_series.ema(price_tensor, n=20)
    ema_50 = time_series.ema(price_tensor, n=50)
    
    returns = time_series.returns(price_tensor)
    volatility = time_series.volatility(price_tensor, n=20, annualize=False)
    
    vwap = time_series.vwap(price_tensor, volume_tensor)
    
    # Z-score of price
    zscore = time_series.zscore(price_tensor, n=50)
    
    return {
        'price': price_tensor,
        'ema_20': ema_20,
        'ema_50': ema_50,
        'returns': returns,
        'volatility': volatility,
        'vwap': vwap,
        'zscore': zscore
    }


def main():
    """Run futures model pipeline example."""
    print("=== Futures Model Pipeline Example ===\n")
    
    # Generate synthetic futures data
    print("Generating synthetic futures data...")
    prices, volumes = generate_futures_data(n_bars=500)
    print(f"Generated {len(prices)} bars\n")
    
    # Create features
    print("Computing quant features...")
    features = create_features(prices, volumes)
    
    # Extract feature vectors (skip initial NaN periods)
    feature_vectors = []
    targets = []
    
    # Use last 400 bars (skip first 100 for warm-up)
    start_idx = 100
    for i in range(start_idx, len(prices) - 1):
        # Feature vector: [ema_20, ema_50, returns, volatility, zscore, vwap]
        # Extract values at index i
        ema20_val = features['ema_20'].data[0][i] if i < len(features['ema_20'].data[0]) else 0.0
        ema50_val = features['ema_50'].data[0][i] if i < len(features['ema_50'].data[0]) else 0.0
        ret_val = features['returns'].data[0][i] if i < len(features['returns'].data[0]) else 0.0
        vol_val = features['volatility'].data[0][i] if i < len(features['volatility'].data[0]) else 0.0
        zscore_val = features['zscore'].data[0][i] if i < len(features['zscore'].data[0]) else 0.0
        vwap_val = features['vwap'].data[0][i] if i < len(features['vwap'].data[0]) else 0.0
        
        feature_vec = [ema20_val, ema50_val, ret_val, vol_val, zscore_val, vwap_val]
        feature_vectors.append(feature_vec)
        
        # Target: next period return
        next_ret = features['returns'].data[0][i + 1] if (i + 1) < len(features['returns'].data[0]) else 0.0
        targets.append(next_ret)
    
    print(f"Created {len(feature_vectors)} feature vectors\n")
    
    # Create model
    print("Initializing model...")
    model = Linear(in_features=6, out_features=1, bias=True)
    print("Model: 6 features -> 1 output (next return prediction)\n")
    
    # Training
    print("Training model...")
    learning_rate = 0.01
    losses = []
    
    # Train on first 200 samples
    n_train = min(200, len(feature_vectors))
    for i in range(n_train):
        x = Tensor([feature_vectors[i]])
        y = Tensor([[targets[i]]])
        
        # Forward
        pred = model.forward(x)
        
        # Loss
        loss = ops.mul(ops.sub(pred, y), ops.sub(pred, y))
        
        # Backward
        if loss.requires_grad:
            loss.backward()
        
        # Get loss
        if isinstance(loss.data[0], list):
            loss_val = loss.data[0][0]
        else:
            loss_val = loss.data[0]
        losses.append(loss_val)
        
        # Clear gradients
        model.zero_grad()
        
        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{n_train}: Loss = {loss_val:.6f}")
    
    print(f"\nTraining completed. Average loss: {sum(losses) / len(losses):.6f}\n")
    
    # Online inference
    print("Performing online inference on remaining data...")
    predictions = []
    actuals = []
    
    for i in range(n_train, len(feature_vectors)):
        x = Tensor([feature_vectors[i]])
        y = Tensor([[targets[i]]])
        
        # Predict
        pred = model.forward(x)
        
        pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
        actual_val = targets[i]
        
        predictions.append(pred_val)
        actuals.append(actual_val)
    
    # Compute prediction error
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    mae = sum(errors) / len(errors) if errors else 0.0
    
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Predictions: {len(predictions)} samples")
    print("\nExample completed!")


if __name__ == "__main__":
    main()

