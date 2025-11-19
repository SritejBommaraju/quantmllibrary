"""
Reproducibility utilities for QuantML.

Provides random seed management, version tracking, and experiment metadata.
"""

import random
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Try to get git info
def get_git_info() -> Dict[str, Optional[str]]:
    """Get git repository information."""
    info = {
        'commit_hash': None,
        'branch': None,
        'is_dirty': None
    }
    
    try:
        import subprocess
        
        # Get commit hash
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            info['commit_hash'] = commit_hash
        except Exception:
            pass
        
        # Get branch
        try:
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            info['branch'] = branch
        except Exception:
            pass
        
        # Check if dirty
        try:
            result = subprocess.run(
                ['git', 'diff', '--quiet'],
                stderr=subprocess.DEVNULL
            )
            info['is_dirty'] = result.returncode != 0
        except Exception:
            pass
    
    except Exception:
        pass
    
    return info


def get_library_version() -> str:
    """Get QuantML library version."""
    try:
        from quantml import __version__
        return __version__
    except ImportError:
        return "unknown"


def set_random_seed(seed: int, use_cuda: bool = False):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        use_cuda: Whether to set CUDA random seed (if available)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if HAS_NUMPY:
        np.random.seed(seed)
    
    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Try to set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information for reproducibility.
    
    Returns:
        Dictionary with environment details
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
        'quantml_version': get_library_version(),
        'git_info': get_git_info()
    }
    
    # Add NumPy version if available
    if HAS_NUMPY:
        info['numpy_version'] = np.__version__
    
    # Try to get other library versions
    for lib_name in ['pandas', 'scipy', 'sklearn']:
        try:
            lib = __import__(lib_name)
            if hasattr(lib, '__version__'):
                info[f'{lib_name}_version'] = lib.__version__
        except ImportError:
            pass
    
    return info


def save_experiment_metadata(
    metadata: Dict[str, Any],
    filepath: str
):
    """
    Save experiment metadata to file.
    
    Args:
        metadata: Metadata dictionary
        filepath: Path to save metadata
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_experiment_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load experiment metadata from file.
    
    Args:
        filepath: Path to metadata file
    
    Returns:
        Metadata dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Create a unique experiment ID.
    
    Args:
        prefix: Prefix for experiment ID
    
    Returns:
        Experiment ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = random.randint(1000, 9999)
    return f"{prefix}_{timestamp}_{random_suffix}"


class ReproducibilityContext:
    """Context manager for reproducible experiments."""
    
    def __init__(self, seed: int, experiment_name: Optional[str] = None):
        """
        Initialize reproducibility context.
        
        Args:
            seed: Random seed
            experiment_name: Optional experiment name
        """
        self.seed = seed
        self.experiment_name = experiment_name
        self.metadata = None
    
    def __enter__(self):
        """Enter context and set random seed."""
        set_random_seed(self.seed)
        self.metadata = get_environment_info()
        self.metadata['random_seed'] = self.seed
        if self.experiment_name:
            self.metadata['experiment_name'] = self.experiment_name
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        return self.metadata.copy() if self.metadata else {}

