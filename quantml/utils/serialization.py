"""
Model serialization utilities for saving and loading models.

This module provides functions to save and load model weights,
as well as full training checkpoints.
"""

import json
import os
from typing import Optional, Dict, Any, List, Union
from quantml.tensor import Tensor

# Try to import NumPy for efficient storage
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def save_model(model, path: str, format: str = 'json') -> None:
    """
    Save model weights to file.
    
    Args:
        model: Model with parameters() method
        path: Path to save weights
        format: 'json' or 'npz' (requires NumPy)
    
    Examples:
        >>> from quantml.models import Linear
        >>> model = Linear(10, 1)
        >>> save_model(model, 'model_weights.json')
    """
    params = model.parameters()
    
    if format == 'json':
        _save_json(params, path)
    elif format == 'npz':
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for npz format")
        _save_npz(params, path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'npz'")


def load_model(model, path: str, format: Optional[str] = None) -> None:
    """
    Load model weights from file.
    
    Args:
        model: Model with parameters() method (weights will be loaded in-place)
        path: Path to load weights from
        format: 'json' or 'npz' (auto-detected from extension if None)
    
    Examples:
        >>> from quantml.models import Linear
        >>> model = Linear(10, 1)
        >>> load_model(model, 'model_weights.json')
    """
    if format is None:
        # Auto-detect from extension
        if path.endswith('.json'):
            format = 'json'
        elif path.endswith('.npz'):
            format = 'npz'
        else:
            format = 'json'  # Default
    
    params = model.parameters()
    
    if format == 'json':
        loaded = _load_json(path)
    elif format == 'npz':
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for npz format")
        loaded = _load_npz(path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Update model parameters
    if len(loaded) != len(params):
        raise ValueError(f"Parameter count mismatch: model has {len(params)}, file has {len(loaded)}")
    
    for i, (param, loaded_data) in enumerate(zip(params, loaded)):
        _update_tensor_data(param, loaded_data)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    path: str,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a full training checkpoint.
    
    Includes model weights, optimizer state, epoch, and optional metrics.
    
    Args:
        model: Model with parameters() method
        optimizer: Optimizer with state to save
        epoch: Current epoch number
        path: Path to save checkpoint
        loss: Optional current loss value
        metrics: Optional dict of metric values
        extra: Optional extra data to save
    
    Examples:
        >>> save_checkpoint(model, optimizer, epoch=10, path='checkpoint.json')
    """
    checkpoint = {
        'epoch': epoch,
        'model_state': _params_to_list(model.parameters()),
        'optimizer_state': _get_optimizer_state(optimizer),
    }
    
    if loss is not None:
        checkpoint['loss'] = loss
    if metrics is not None:
        checkpoint['metrics'] = metrics
    if extra is not None:
        checkpoint['extra'] = extra
    
    # Save as JSON
    with open(path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(
    model,
    optimizer,
    path: str
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Restores model weights and optimizer state.
    
    Args:
        model: Model with parameters() method
        optimizer: Optimizer to restore state to
        path: Path to load checkpoint from
    
    Returns:
        Dict with 'epoch', 'loss', 'metrics', and any extra data
    
    Examples:
        >>> info = load_checkpoint(model, optimizer, 'checkpoint.json')
        >>> print(f"Resuming from epoch {info['epoch']}")
    """
    with open(path, 'r') as f:
        checkpoint = json.load(f)
    
    # Restore model weights
    params = model.parameters()
    loaded = checkpoint['model_state']
    
    if len(loaded) != len(params):
        raise ValueError(f"Parameter count mismatch: model has {len(params)}, checkpoint has {len(loaded)}")
    
    for param, loaded_data in zip(params, loaded):
        _update_tensor_data(param, loaded_data)
    
    # Restore optimizer state
    if 'optimizer_state' in checkpoint:
        _set_optimizer_state(optimizer, checkpoint['optimizer_state'])
    
    # Return info
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss'),
        'metrics': checkpoint.get('metrics'),
        'extra': checkpoint.get('extra')
    }


def get_model_state_dict(model) -> Dict[str, List]:
    """
    Get model state as a dictionary.
    
    Args:
        model: Model with parameters() method
    
    Returns:
        Dict mapping parameter names to their data
    """
    params = model.parameters()
    state = {}
    for i, param in enumerate(params):
        name = f"param_{i}"
        if hasattr(param, 'data'):
            state[name] = _tensor_to_list(param)
        else:
            state[name] = param
    return state


def set_model_state_dict(model, state_dict: Dict[str, List]) -> None:
    """
    Set model state from a dictionary.
    
    Args:
        model: Model with parameters() method
        state_dict: Dict mapping parameter names to their data
    """
    params = model.parameters()
    for i, param in enumerate(params):
        name = f"param_{i}"
        if name in state_dict:
            _update_tensor_data(param, state_dict[name])


# ============================================================================
# Private helper functions
# ============================================================================

def _save_json(params: List[Tensor], path: str) -> None:
    """Save parameters to JSON file."""
    data = _params_to_list(params)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_json(path: str) -> List:
    """Load parameters from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def _save_npz(params: List[Tensor], path: str) -> None:
    """Save parameters to NumPy npz file."""
    arrays = {}
    for i, param in enumerate(params):
        arr = _tensor_to_numpy(param)
        arrays[f'param_{i}'] = arr
    np.savez(path, **arrays)


def _load_npz(path: str) -> List:
    """Load parameters from NumPy npz file."""
    data = np.load(path)
    # Sort by parameter index
    keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))
    return [data[k].tolist() for k in keys]


def _params_to_list(params: List[Tensor]) -> List:
    """Convert list of Tensors to list of nested lists."""
    return [_tensor_to_list(p) for p in params]


def _tensor_to_list(t: Tensor) -> List:
    """Convert Tensor to nested list."""
    data = t.data
    if isinstance(data, list):
        return data
    # NumPy array
    try:
        return data.tolist()
    except AttributeError:
        return list(data)


def _tensor_to_numpy(t: Tensor):
    """Convert Tensor to NumPy array."""
    if hasattr(t, 'numpy') and t.numpy is not None:
        return t.numpy
    data = t.data
    if isinstance(data, list):
        return np.array(data)
    return data


def _update_tensor_data(param: Tensor, data) -> None:
    """Update tensor data in-place."""
    if isinstance(data, list):
        param._data_list = data
    else:
        # NumPy array
        param._data_list = data.tolist() if hasattr(data, 'tolist') else list(data)

    # Clear cached numpy array so it gets rebuilt from _data_list
    param._np_array = None


def _get_optimizer_state(optimizer) -> Dict[str, Any]:
    """Extract optimizer state for serialization."""
    state = {
        'lr': getattr(optimizer, 'lr', None),
        'step': getattr(optimizer, '_step', 0),
    }
    
    # Handle Adam-like optimizers with momentum buffers
    if hasattr(optimizer, '_m'):
        state['m'] = [_tensor_to_list(m) if hasattr(m, 'data') else m 
                      for m in optimizer._m]
    if hasattr(optimizer, '_v'):
        state['v'] = [_tensor_to_list(v) if hasattr(v, 'data') else v 
                      for v in optimizer._v]
    
    # Handle SGD momentum
    if hasattr(optimizer, '_velocity'):
        state['velocity'] = [_tensor_to_list(v) if hasattr(v, 'data') else v 
                            for v in optimizer._velocity]
    
    return state


def _set_optimizer_state(optimizer, state: Dict[str, Any]) -> None:
    """Restore optimizer state from serialized data."""
    if 'step' in state:
        optimizer._step = state['step']
    
    # Handle Adam-like optimizers
    if 'm' in state and hasattr(optimizer, '_m'):
        for i, m_data in enumerate(state['m']):
            if i < len(optimizer._m):
                _update_tensor_data(optimizer._m[i], m_data)
    
    if 'v' in state and hasattr(optimizer, '_v'):
        for i, v_data in enumerate(state['v']):
            if i < len(optimizer._v):
                _update_tensor_data(optimizer._v[i], v_data)
    
    # Handle SGD momentum
    if 'velocity' in state and hasattr(optimizer, '_velocity'):
        for i, v_data in enumerate(state['velocity']):
            if i < len(optimizer._velocity):
                _update_tensor_data(optimizer._velocity[i], v_data)
