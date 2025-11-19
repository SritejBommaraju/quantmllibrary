"""
Example: Streaming Training

This example demonstrates training a model on streaming tick-level data
using StreamingTensor and per-tick updates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml import ops
from quantml.streaming import StreamingTensor
from quantml.online import StreamingDataset, per_tick_training_step


def generate_tick_data(n_ticks=200):
    """Generate synthetic tick-level market data."""
    prices = []
    volumes = []
    base_price = 100.0
    
    for i in range(n_ticks):
        # Random walk with drift
        change = (i % 20 - 10) * 0.1
        price = base_price + change + (i * 0.01)
        prices.append(price)
        volumes.append(10.0 + (i % 5) * 2.0)
        base_price = price
    
    return prices, volumes


def main():
    """Run streaming training example."""
    print("=== Streaming Training Example ===\n")
    
    # Create model to predict next price from current price and volume
    model = Linear(in_features=2, out_features=1, bias=True)
    
    # Generate synthetic tick data
    prices, volumes = generate_tick_data(n_ticks=200)
    
    # Create streaming datasets
    feature_stream = StreamingTensor(max_size=1000)
    target_stream = StreamingTensor(max_size=1000)
    
    # Prepare features: [price, volume]
    # Target: next price
    print("Preparing streaming data...")
    for i in range(len(prices) - 1):
        feature = [prices[i], volumes[i]]
        target = prices[i + 1]
        feature_stream.append(feature)
        target_stream.append(target)
    
    print(f"Created {len(feature_stream)} samples\n")
    
    # Create streaming dataset
    dataset = StreamingDataset(feature_stream, target_stream)
    
    # Training loop
    print("Training on streaming ticks...")
    learning_rate = 0.001
    losses = []
    
    for i in range(min(50, len(feature_stream))):  # Train on first 50 ticks
        # Get sample
        x_batch, y_batch = dataset.get_batch(size=1)
        
        # Ensure proper shape
        if isinstance(x_batch.data[0], list):
            x_val = x_batch.data[0]
        else:
            x_val = x_batch.data
        
        if isinstance(y_batch.data[0], list):
            y_val = y_batch.data[0][0]
        else:
            y_val = y_batch.data[0]
        
        x_tensor = Tensor([x_val])
        y_tensor = Tensor([[y_val]])
        
        # Forward pass
        pred = model.forward(x_tensor)
        
        # Loss
        loss = ops.mul(ops.sub(pred, y_tensor), ops.sub(pred, y_tensor))
        
        # Backward
        if loss.requires_grad:
            loss.backward()
        
        # Get loss value
        if isinstance(loss.data[0], list):
            loss_val = loss.data[0][0]
        else:
            loss_val = loss.data[0]
        losses.append(loss_val)
        
        # Clear gradients for next step
        model.zero_grad()
        
        if (i + 1) % 10 == 0:
            print(f"Tick {i+1}: Loss = {loss_val:.4f}, Pred = {pred.data[0][0]:.2f}, Target = {y_val:.2f}")
    
    print(f"\nAverage loss: {sum(losses) / len(losses):.4f}")
    print("\nExample completed!")


if __name__ == "__main__":
    main()

