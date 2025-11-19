"""
Configuration management for QuantML experiments.

Supports YAML/JSON config files and command-line argument integration.
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import argparse

# Try to import YAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    instrument: str = "ES"  # MES, ES, MNQ, NQ, etc.
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    data_source: str = "csv"  # csv, database, api
    data_path: Optional[str] = None
    frequency: str = "1min"  # 1min, 5min, daily
    validate_data: bool = True
    handle_missing: str = "forward_fill"  # forward_fill, drop, interpolate
    cache_features: bool = True
    feature_cache_path: str = "./cache/features"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    enabled_features: List[str] = field(default_factory=lambda: [
        "lagged_price",
        "rolling_mean",
        "rolling_std",
        "returns",
        "volatility"
    ])
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    rolling_windows: List[int] = field(default_factory=lambda: [20, 50])
    normalize: bool = True
    normalization_method: str = "zscore"  # zscore, minmax, robust
    alpha_factors: Dict[str, Any] = field(default_factory=lambda: {
        "momentum": {"enabled": True, "lookback": 20},
        "mean_reversion": {"enabled": True, "window": 20},
        "volatility": {"enabled": True, "window": 20}
    })


@dataclass
class ModelConfig:
    """Model architecture and hyperparameters."""
    model_type: str = "Linear"  # Linear, SimpleRNN, TCN
    in_features: int = 10
    out_features: int = 1
    hidden_size: Optional[int] = None
    bias: bool = True
    dropout: float = 0.0
    activation: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    optimizer: str = "Adam"  # SGD, Adam, RMSProp, etc.
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    loss_function: str = "mse_loss"  # mse_loss, sharpe_loss, etc.
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.0001
    })
    gradient_clipping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "max_norm": 1.0
    })
    scheduler: Optional[Dict[str, Any]] = None
    walk_forward: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "train_size": 500,
        "test_size": 100,
        "window_type": "expanding"  # expanding, rolling
    })


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default_experiment"
    description: str = ""
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./experiments"
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Handle nested configs
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'features' in data and isinstance(data['features'], dict):
            data['features'] = FeatureConfig(**data['features'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate dates
        try:
            from datetime import datetime
            datetime.strptime(self.data.start_date, "%Y-%m-%d")
            datetime.strptime(self.data.end_date, "%Y-%m-%d")
        except ValueError:
            errors.append("Invalid date format. Use YYYY-MM-DD")
        
        # Validate learning rate
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        # Validate batch size
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Validate model features
        if self.model.in_features <= 0:
            errors.append("Model input features must be positive")
        
        return errors


class Config:
    """Main configuration class."""
    
    @staticmethod
    def load_yaml(filepath: str) -> ExperimentConfig:
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return ExperimentConfig.from_dict(data)
    
    @staticmethod
    def load_json(filepath: str) -> ExperimentConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return ExperimentConfig.from_dict(data)
    
    @staticmethod
    def save_yaml(config: ExperimentConfig, filepath: str):
        """Save configuration to YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pyyaml")
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def save_json(config: ExperimentConfig, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)


def load_config(filepath: str) -> ExperimentConfig:
    """Load configuration from file (auto-detect YAML/JSON)."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.yaml', '.yml']:
        return Config.load_yaml(filepath)
    elif ext == '.json':
        return Config.load_json(filepath)
    else:
        raise ValueError(f"Unsupported config file format: {ext}. Use .yaml, .yml, or .json")


def save_config(config: ExperimentConfig, filepath: str):
    """Save configuration to file (auto-detect YAML/JSON)."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.yaml', '.yml']:
        Config.save_yaml(config, filepath)
    elif ext == '.json':
        Config.save_json(config, filepath)
    else:
        raise ValueError(f"Unsupported config file format: {ext}. Use .yaml, .yml, or .json")


def create_argparser() -> argparse.ArgumentParser:
    """Create argument parser with common config options."""
    parser = argparse.ArgumentParser(description='QuantML Experiment Runner')
    
    parser.add_argument('--config', type=str, help='Path to config file (YAML/JSON)')
    parser.add_argument('--instrument', type=str, help='Trading instrument (ES, MES, NQ, MNQ)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-type', type=str, help='Model type (Linear, SimpleRNN, TCN)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--random-seed', type=int, help='Random seed')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    return parser


def merge_config_with_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Merge command-line arguments into config."""
    if args.instrument:
        config.data.instrument = args.instrument
    if args.start_date:
        config.data.start_date = args.start_date
    if args.end_date:
        config.data.end_date = args.end_date
    if args.model_type:
        config.model.model_type = args.model_type
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.random_seed:
        config.random_seed = args.random_seed
    if args.output_dir:
        config.output_dir = args.output_dir
    
    return config

