"""
Feature store for caching computed features.

Uses Parquet format for efficient storage and loading of large feature datasets.
"""

import os
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Try to import pandas and pyarrow
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    pq = None


class FeatureStore:
    """Feature caching system using Parquet format."""
    
    def __init__(
        self,
        cache_dir: str = "./cache/features",
        use_parquet: bool = True
    ):
        """
        Initialize feature store.
        
        Args:
            cache_dir: Directory for cached features
            use_parquet: Whether to use Parquet format (requires pyarrow)
        """
        self.cache_dir = cache_dir
        self.use_parquet = use_parquet and HAS_PARQUET
        
        if not self.use_parquet and not HAS_PANDAS:
            raise ImportError(
                "Either pandas or pyarrow required. "
                "Install with: pip install pandas pyarrow"
            )
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(
        self,
        instrument: str,
        start_date: str,
        end_date: str,
        feature_config: Dict[str, Any]
    ) -> str:
        """Generate cache key from configuration."""
        key_data = {
            'instrument': instrument,
            'start_date': start_date,
            'end_date': end_date,
            'features': feature_config
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        if self.use_parquet:
            return os.path.join(self.cache_dir, f"{cache_key}.parquet")
        else:
            return os.path.join(self.cache_dir, f"{cache_key}.csv")
    
    def _get_metadata_path(self, cache_key: str) -> str:
        """Get metadata file path."""
        return os.path.join(self.cache_dir, f"{cache_key}_metadata.json")
    
    def save_features(
        self,
        features: List[List[float]],
        instrument: str,
        start_date: str,
        end_date: str,
        feature_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save features to cache.
        
        Args:
            features: List of feature vectors
            instrument: Trading instrument
            start_date: Start date
            end_date: End date
            feature_config: Feature configuration
            metadata: Optional metadata
        
        Returns:
            Cache key
        """
        cache_key = self._generate_cache_key(
            instrument, start_date, end_date, feature_config
        )
        
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        # Convert to DataFrame
        if not HAS_PANDAS:
            raise ImportError("pandas required for feature storage")
        
        df = pd.DataFrame(features)
        
        # Save features
        if self.use_parquet:
            df.to_parquet(cache_path, compression='snappy', index=False)
        else:
            df.to_csv(cache_path, index=False)
        
        # Save metadata
        meta = {
            'cache_key': cache_key,
            'instrument': instrument,
            'start_date': start_date,
            'end_date': end_date,
            'feature_config': feature_config,
            'num_features': len(features[0]) if features else 0,
            'num_samples': len(features),
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        return cache_key
    
    def load_features(self, cache_key: str) -> tuple:
        """
        Load features from cache.
        
        Args:
            cache_key: Cache key
        
        Returns:
            Tuple of (features, metadata)
        """
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found: {cache_key}")
        
        # Load features
        if not HAS_PANDAS:
            raise ImportError("pandas required for feature loading")
        
        if self.use_parquet:
            df = pd.read_parquet(cache_path)
        else:
            df = pd.read_csv(cache_path)
        
        features = df.values.tolist()
        
        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return features, metadata
    
    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists."""
        cache_path = self._get_cache_path(cache_key)
        return os.path.exists(cache_path)
    
    def invalidate_cache(
        self,
        instrument: Optional[str] = None,
        cache_key: Optional[str] = None
    ):
        """
        Invalidate cache entries.
        
        Args:
            instrument: Invalidate all caches for this instrument
            cache_key: Invalidate specific cache key
        """
        if cache_key:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        
        elif instrument:
            # Remove all caches for instrument
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_metadata.json'):
                    metadata_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(metadata_path, 'r') as f:
                            meta = json.load(f)
                            if meta.get('instrument') == instrument:
                                cache_key = meta.get('cache_key')
                                if cache_key:
                                    self.invalidate_cache(cache_key=cache_key)
                    except Exception:
                        pass
    
    def list_caches(self) -> List[Dict[str, Any]]:
        """List all cached features."""
        caches = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_metadata.json'):
                metadata_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                        caches.append(meta)
                except Exception:
                    pass
        
        return caches

