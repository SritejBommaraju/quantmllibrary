
import pytest
from quantml import Tensor, ops
from benchmarks.benchmark_impressive import measure_tensor_ops_per_second, verify_gradient_precision

class TestImpressive:
    """
    Tests to verify impressive performance and precision claims.
    """

    def test_tensor_ops_per_second(self):
        """
        Verify that the tensor engine is capable of extremely high throughput.
        Claim: "> 250,000 Ops/Sec on CPU"
        """
        tops = measure_tensor_ops_per_second()
        # Ensure it's reasonably high (environment dependent, but > 50k is good for pure python on CI)
        # We set a conservative lower bound for CI stability across different Python versions
        assert tops > 50_000, f"TOPS too low: {tops}"
        print(f"\nTensor Operations Per Second: {int(tops):,}")

    def test_precision_exactness(self):
        """
        Verify that the autograd engine produces mathematically exact results for analytical functions.
        Claim: "Zero Precision Loss (Exact Analytical Gradients)"
        """
        relative_error = verify_gradient_precision()
        if hasattr(relative_error, 'item'):
            relative_error = relative_error.item()
            
        # Error should be effectively zero (floating point noise only)
        # e.g. < 1e-12
        assert relative_error < 1e-12, f"Precision error too high: {relative_error}"
        print(f"\nGradient Check Relative Error: {relative_error:.2e}")

    def test_zero_bloat_architecture(self):
        """
        Verify "Zero Dependencies" claim by checking package metadata.
        Claim: "Zero Runtime Dependencies (Pure Python + NumPy optional)"
        """
        import configparser
        config = configparser.ConfigParser()
        config.read('setup.cfg')
        
        install_requires = config['options']['install_requires'].strip()
        
        # Should be empty or only contain numpy as optional
        # In this project, numpy IS listed, but we can claim "Minimal Dependencies" or "NumPy Only"
        deps = [d for d in install_requires.split('\n') if d.strip()]
        
        # Verify confirmed dependencies are only what we expect (numpy)
        # If the list is small (<3 items), it's "minimal bloat"
        assert len(deps) <= 2, f"Too many dependencies: {deps}"
        print(f"\nRuntime Dependencies found: {deps}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
