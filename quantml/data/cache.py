"""
Cache management utilities.
"""

from typing import Optional, Callable, Any
from functools import wraps
import hashlib
import json
import os

from quantml.data.feature_store import FeatureStore


def cached_features(
    cache_dir: str = "./cache/features",
    use_cache: bool = True
):
    """
    Decorator to cache feature computation results.
    
    Args:
        cache_dir: Cache directory
        use_cache: Whether to use cache
    
    Example:
        @cached_features(cache_dir="./cache")
        def compute_features(data):
            # Expensive computation
            return features
    """
    def decorator(func: Callable) -> Callable:
        store = FeatureStore(cache_dir=cache_dir)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)
            
            # Generate cache key from function arguments
            cache_key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': json.dumps(kwargs, sort_keys=True)
            }
            cache_key_str = json.dumps(cache_key_data, sort_keys=True)
            cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
            
            # Check cache
            if store.cache_exists(cache_key):
                features, _ = store.load_features(cache_key)
                return features
            
            # Compute features
            features = func(*args, **kwargs)
            
            # Save to cache
            # Extract metadata from kwargs if available
            instrument = kwargs.get('instrument', 'unknown')
            start_date = kwargs.get('start_date', 'unknown')
            end_date = kwargs.get('end_date', 'unknown')
            feature_config = kwargs.get('feature_config', {})
            
            store.save_features(
                features,
                instrument,
                start_date,
                end_date,
                feature_config
            )
            
            return features
        
        return wrapper
    return decorator


class CacheManager:
    """Cache manager for feature computation."""
    
    def __init__(self, cache_dir: str = "./cache/features"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Cache directory
        """
        self.store = FeatureStore(cache_dir=cache_dir)
        self.cache_dir = cache_dir
    
    def get_or_compute(
        self,
        cache_key: str,
        compute_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Get from cache or compute.
        
        Args:
            cache_key: Cache key
            compute_fn: Function to compute if not cached
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function
        
        Returns:
            Cached or computed result
        """
        if self.store.cache_exists(cache_key):
            features, _ = self.store.load_features(cache_key)
            return features
        
        # Compute
        result = compute_fn(*args, **kwargs)
        
        # Save to cache if result is features
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):  # List of feature vectors
                instrument = kwargs.get('instrument', 'unknown')
                start_date = kwargs.get('start_date', 'unknown')
                end_date = kwargs.get('end_date', 'unknown')
                feature_config = kwargs.get('feature_config', {})
                
                self.store.save_features(
                    result,
                    instrument,
                    start_date,
                    end_date,
                    feature_config
                )
        
        return result
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            pattern: Optional pattern to match (e.g., instrument name)
        """
        if pattern:
            self.store.invalidate_cache(instrument=pattern)
        else:
            # Clear all
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)

