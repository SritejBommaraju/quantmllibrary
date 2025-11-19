"""
Memory optimization utilities for large datasets.
"""

from typing import List, Dict, Any, Optional
import sys

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def optimize_dtypes(df: Any) -> Any:
    """
    Optimize pandas DataFrame dtypes to reduce memory usage.
    
    Args:
        df: Pandas DataFrame
    
    Returns:
        DataFrame with optimized dtypes
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required. Install with: pip install pandas")
    
    if not isinstance(df, pd.DataFrame):
        return df
    
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns (strings)
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        if num_unique / num_total < 0.5:  # Low cardinality
            df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    reduction = (1 - new_memory / original_memory) * 100
    
    return df


def chunked_process(
    data: List,
    chunk_size: int,
    process_fn: callable,
    *args,
    **kwargs
) -> List:
    """
    Process data in chunks to reduce memory usage.
    
    Args:
        data: List of data to process
        chunk_size: Size of each chunk
        process_fn: Function to process each chunk
        *args: Arguments for process function
        **kwargs: Keyword arguments for process function
    
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_result = process_fn(chunk, *args, **kwargs)
        results.extend(chunk_result)
    
    return results


def estimate_memory_usage(data: Any) -> Dict[str, Any]:
    """
    Estimate memory usage of data structure.
    
    Args:
        data: Data structure (list, dict, DataFrame, etc.)
    
    Returns:
        Dictionary with memory usage information
    """
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        memory_info = {
            'total_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'per_column': {
                col: size / 1024 / 1024
                for col, size in data.memory_usage(deep=True).items()
            },
            'num_rows': len(data),
            'num_columns': len(data.columns)
        }
        return memory_info
    
    elif isinstance(data, list):
        # Rough estimate
        if len(data) == 0:
            return {'total_mb': 0, 'num_items': 0}
        
        # Estimate size of first item
        sample_size = sys.getsizeof(data[0])
        if isinstance(data[0], list):
            sample_size = sum(sys.getsizeof(item) for item in data[0])
        
        total_size = len(data) * sample_size
        memory_info = {
            'total_mb': total_size / 1024 / 1024,
            'num_items': len(data),
            'estimated_item_size_bytes': sample_size
        }
        return memory_info
    
    else:
        size_bytes = sys.getsizeof(data)
        return {
            'total_mb': size_bytes / 1024 / 1024,
            'size_bytes': size_bytes
        }


class StreamingDataProcessor:
    """Process large datasets in streaming fashion."""
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize streaming processor.
        
        Args:
            chunk_size: Size of each processing chunk
        """
        self.chunk_size = chunk_size
    
    def process_file(
        self,
        filepath: str,
        process_fn: callable,
        *args,
        **kwargs
    ) -> List:
        """
        Process file in chunks.
        
        Args:
            filepath: Path to data file
            process_fn: Function to process each chunk
            *args: Arguments for process function
            **kwargs: Keyword arguments for process function
        
        Returns:
            List of processed results
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for file processing")
        
        results = []
        
        # Read file in chunks
        chunk_iterator = pd.read_csv(
            filepath,
            chunksize=self.chunk_size
        )
        
        for chunk in chunk_iterator:
            chunk_result = process_fn(chunk, *args, **kwargs)
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)
        
        return results
    
    def process_iterable(
        self,
        data_iterable,
        process_fn: callable,
        *args,
        **kwargs
    ) -> List:
        """
        Process iterable in chunks.
        
        Args:
            data_iterable: Iterable data source
            process_fn: Function to process each chunk
            *args: Arguments for process function
            **kwargs: Keyword arguments for process function
        
        Returns:
            List of processed results
        """
        results = []
        chunk = []
        
        for item in data_iterable:
            chunk.append(item)
            
            if len(chunk) >= self.chunk_size:
                chunk_result = process_fn(chunk, *args, **kwargs)
                if isinstance(chunk_result, list):
                    results.extend(chunk_result)
                else:
                    results.append(chunk_result)
                chunk = []
        
        # Process remaining items
        if chunk:
            chunk_result = process_fn(chunk, *args, **kwargs)
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)
        
        return results

