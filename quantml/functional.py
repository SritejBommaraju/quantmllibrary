"""
Functional API for QuantML operations.

This module provides a convenient functional interface to all operations,
similar to PyTorch's functional API. Use this for a cleaner API when
composing operations.
"""

from typing import Union, Optional
from quantml.tensor import Tensor
from quantml import ops

# Re-export all operations with F. prefix
add = ops.add
sub = ops.sub
mul = ops.mul
div = ops.div
pow = ops.pow
matmul = ops.matmul
dot = ops.dot
sum = ops.sum
mean = ops.mean
std = ops.std
relu = ops.relu
sigmoid = ops.sigmoid
tanh = ops.tanh
abs = ops.abs
maximum = ops.maximum
transpose = ops.transpose
select = ops.select
stack = ops.stack

__all__ = [
    'add', 'sub', 'mul', 'div', 'pow',
    'matmul', 'dot',
    'sum', 'mean', 'std',
    'relu', 'sigmoid', 'tanh',
    'abs', 'maximum',
    'transpose', 'select', 'stack'
]

