"""
Enhanced experiment tracker with dataset versioning and results comparison.
"""

import os
import json
import csv
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """
    Enhanced experiment tracker.
    
    Tracks:
    - Dataset version
    - Feature set version
    - Hyperparameters
    - Random seed
    - Model architecture
    - Run date/time with git hash
    - Results
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment data
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {}
        self.results = {}
        self.start_time = datetime.now()
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.metadata['config'] = config
        self.metadata['random_seed'] = config.get('random_seed')
        self.metadata['model_type'] = config.get('model', {}).get('model_type')
        self.metadata['optimizer'] = config.get('training', {}).get('optimizer')
        self.metadata['learning_rate'] = config.get('training', {}).get('learning_rate')
    
    def log_dataset_version(self, dataset_path: str, version: Optional[str] = None):
        """
        Log dataset version.
        
        Args:
            dataset_path: Path to dataset
            version: Optional version string (if None, uses file modification time)
        """
        if version is None:
            # Use file modification time as version
            if os.path.exists(dataset_path):
                mtime = os.path.getmtime(dataset_path)
                version = datetime.fromtimestamp(mtime).isoformat()
            else:
                version = "unknown"
        
        self.metadata['dataset_path'] = dataset_path
        self.metadata['dataset_version'] = version
    
    def log_feature_set(self, features: List[str], version: Optional[str] = None):
        """
        Log feature set.
        
        Args:
            features: List of feature names
            version: Optional version string (hash of feature list)
        """
        if version is None:
            # Create hash from feature list
            import hashlib
            feature_str = ','.join(sorted(features))
            version = hashlib.md5(feature_str.encode()).hexdigest()[:8]
        
        self.metadata['features'] = features
        self.metadata['feature_set_version'] = version
    
    def log_model_architecture(self, model_config: Dict[str, Any]):
        """Log model architecture details."""
        self.metadata['model_architecture'] = model_config
    
    def log_git_hash(self):
        """Log current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            if result.returncode == 0:
                self.metadata['git_hash'] = result.stdout.strip()
        except Exception:
            self.metadata['git_hash'] = "unknown"
    
    def log_results(self, results: Dict[str, Any]):
        """Log experiment results."""
        self.results = results
        self.metadata['results'] = results
    
    def save(self):
        """Save experiment data to disk."""
        self.metadata['start_time'] = self.start_time.isoformat()
        self.metadata['end_time'] = datetime.now().isoformat()
        
        # Save metadata
        metadata_path = self.experiment_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save results
        if self.results:
            results_path = self.experiment_dir / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        
        # Append to CSV log
        self._append_to_csv_log()
    
    def _append_to_csv_log(self):
        """Append experiment to CSV log for comparison."""
        csv_path = self.experiment_dir.parent / 'experiments_log.csv'
        
        # Check if CSV exists
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id',
                'start_time',
                'model_type',
                'optimizer',
                'learning_rate',
                'dataset_version',
                'feature_set_version',
                'git_hash',
                'ic',
                'sharpe_ratio',
                'total_return',
                'experiment_dir'
            ])
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                'experiment_id': self.experiment_dir.name,
                'start_time': self.metadata.get('start_time', ''),
                'model_type': self.metadata.get('model_type', ''),
                'optimizer': self.metadata.get('optimizer', ''),
                'learning_rate': self.metadata.get('learning_rate', ''),
                'dataset_version': self.metadata.get('dataset_version', ''),
                'feature_set_version': self.metadata.get('feature_set_version', ''),
                'git_hash': self.metadata.get('git_hash', ''),
                'ic': self.results.get('alpha_metrics', {}).get('ic', ''),
                'sharpe_ratio': self.results.get('backtest_results', {}).get('sharpe_ratio', ''),
                'total_return': self.results.get('backtest_results', {}).get('total_return', ''),
                'experiment_dir': str(self.experiment_dir)
            }
            
            writer.writerow(row)


def compare_experiments(experiment_dirs: List[str]) -> Dict[str, Any]:
    """
    Compare multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
    
    Returns:
        Comparison dictionary
    """
    experiments = []
    
    for exp_dir in experiment_dirs:
        metadata_path = Path(exp_dir) / 'metadata.json'
        results_path = Path(exp_dir) / 'results.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}
        
        experiments.append({
            'dir': exp_dir,
            'metadata': metadata,
            'results': results
        })
    
    # Create comparison
    comparison = {
        'n_experiments': len(experiments),
        'experiments': []
    }
    
    for exp in experiments:
        comparison['experiments'].append({
            'dir': exp['dir'],
            'model_type': exp['metadata'].get('model_type'),
            'optimizer': exp['metadata'].get('optimizer'),
            'ic': exp['results'].get('alpha_metrics', {}).get('ic'),
            'sharpe': exp['results'].get('backtest_results', {}).get('sharpe_ratio'),
            'return': exp['results'].get('backtest_results', {}).get('total_return')
        })
    
    return comparison

