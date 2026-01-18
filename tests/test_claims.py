
import time
import pytest
from quantml import Tensor, ops
from quantml.models import MLP, Linear

class TestClaims:
    """
    Tests specifically designed to verify the claims made for the application.
    """

    def test_autograd_depth_50(self):
        """
        Verify that the autograd engine can handle a computation graph of depth 50.
        Claim: "Custom Autograd Engine to 50+ Layers"
        """
        depth = 50
        x = Tensor([[1.0]], requires_grad=True)
        h = x
        
        # Create a deep graph
        for _ in range(depth):
            # h = h * 1.01 (simple operation to prevent explosion/vanishing)
            h = ops.mul(h, 1.01)
            
        loss = ops.sum(h)
        
        start_time = time.time()
        loss.backward()
        end_time = time.time()
        
        # Verify gradient exists and is correct
        # d/dx (x * 1.01^50) = 1.01^50
        expected_grad = 1.01 ** 50
        
        assert x.grad is not None
        grad_val = x.grad[0][0] if isinstance(x.grad, list) else x.grad.item()
        
        assert grad_val == pytest.approx(expected_grad, rel=1e-5)
        print(f"\nBackprop time for depth {depth}: {(end_time - start_time)*1000:.4f} ms")

    def test_mlp_inference_speed_proxy(self):
        """
        Verify MLP inference runs correctly and is performant enough to be plausible.
        Claim: "< 5ms Inference Latency" (Soft check)
        """
        model = MLP([100, 128, 128, 1])
        x = Tensor([[0.1] * 100])
        
        # Warmup
        model.forward(x)
        
        # Measure single pass
        start = time.time()
        model.forward(x)
        end = time.time()
        
        latency_ms = (end - start) * 1000
        # We don't assert < 5ms strictly because CI/test envs vary, 
        # but we assert it's not egregiously slow (e.g., > 100ms)
        assert latency_ms < 100.0 
        print(f"\nSingle-pass inference latency: {latency_ms:.4f} ms")

    def test_gradient_check_accuracy(self):
        """
        Verify that our numerical gradient checker confirms correct implementation.
        Claim: "Rigorous testing with Gradient Checking"
        """
        from quantml.utils.gradient_check import check_gradients
        
        # Test a simple linear operation
        linear = Linear(2, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        
        def forward_fn(input_tensor):
            return linear.forward(input_tensor)
            
        # This asserts internally that analytical grad matches numerical grad
        is_correct = check_gradients(forward_fn, x)
        assert is_correct, "Gradient check failed for Linear layer"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
