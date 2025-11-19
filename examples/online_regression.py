"""
Example: Online Regression

This example demonstrates online learning with streaming data,
where a linear model is updated incrementally as new data arrives.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml import ops
from quantml.streaming import StreamingTensor
from quantml.online import per_tick_training_step, incremental_update


def main():
    """Run online regression example."""
    print("=== Online Regression Example ===\n")
    
    # Create a simple linear model: y = 2*x + 1 + noise
    model = Linear(in_features=1, out_features=1, bias=True)
    
    # Initialize with small random weights
    # In practice, you'd use proper initialization
    print("Model initialized")
    print(f"Weight shape: {model.weight.shape}")
    print(f"Bias shape: {model.bias_param.shape if model.bias else 'None'}\n")
    
    # Create streaming data
    x_stream = StreamingTensor(max_size=1000)
    y_stream = StreamingTensor(max_size=1000)
    
    # Generate synthetic streaming data
    true_weight = 2.0
    true_bias = 1.0
    learning_rate = 0.01
    
    print("Training on streaming data...")
    print("True relationship: y = 2.0 * x + 1.0\n")
    
    losses = []
    for i in range(100):
        # Generate new data point
        x_val = float(i) / 10.0
        noise = (i % 10 - 5) * 0.1  # Simple noise pattern
        y_val = true_weight * x_val + true_bias + noise
        
        # Append to streams
        x_stream.append(x_val)
        y_stream.append(y_val)
        
        # Get current sample
        x_tensor = Tensor([[x_val]])
        y_tensor = Tensor([[y_val]])
        
        # Forward pass
        pred = model.forward(x_tensor)
        
        # Compute loss (MSE)
        loss = ops.mul(ops.sub(pred, y_tensor), ops.sub(pred, y_tensor))
        
        # Backward pass
        if loss.requires_grad:
            loss.backward()
        
        # Online update
        params = model.parameters()
        grads = [p.grad for p in params if p.grad is not None]
        
        if len(grads) == len(params):
            updated = incremental_update(params, grads, learning_rate)
            # In a real implementation, we'd update model parameters in-place
            # For demonstration, we show the pattern
        
        # Store loss
        if isinstance(loss.data[0], list):
            loss_val = loss.data[0][0]
        else:
            loss_val = loss.data[0]
        losses.append(loss_val)
        
        # Print progress every 10 steps
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: Loss = {loss_val:.4f}")
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Average loss (last 10): {sum(losses[-10:]) / 10:.4f}")
    print("\nExample completed!")


if __name__ == "__main__":
    main()

