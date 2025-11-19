"""
CSV/JSON logger for experiment tracking.
"""

import csv
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


class CSVExperimentLogger:
    """
    CSV-based experiment logger.
    
    Logs experiments to a CSV file for easy comparison and analysis.
    """
    
    def __init__(self, log_file: str = "experiments_log.csv"):
        """
        Initialize CSV logger.
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = Path(log_file)
        self.fieldnames = [
            'timestamp',
            'experiment_id',
            'model_type',
            'optimizer',
            'learning_rate',
            'batch_size',
            'epochs',
            'dataset_version',
            'feature_set',
            'random_seed',
            'ic',
            'rank_ic',
            'sharpe_ratio',
            'total_return',
            'max_drawdown',
            'win_rate',
            'n_trades',
            'notes'
        ]
    
    def log_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        notes: Optional[str] = None
    ):
        """
        Log an experiment to CSV.
        
        Args:
            experiment_id: Unique experiment identifier
            config: Experiment configuration
            results: Experiment results
            notes: Optional notes
        """
        file_exists = self.log_file.exists()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                'timestamp': datetime.now().isoformat(),
                'experiment_id': experiment_id,
                'model_type': config.get('model', {}).get('model_type', ''),
                'optimizer': config.get('training', {}).get('optimizer', ''),
                'learning_rate': config.get('training', {}).get('learning_rate', ''),
                'batch_size': config.get('training', {}).get('batch_size', ''),
                'epochs': config.get('training', {}).get('epochs', ''),
                'dataset_version': config.get('data', {}).get('dataset_version', ''),
                'feature_set': ','.join(config.get('features', {}).get('enabled_features', [])),
                'random_seed': config.get('random_seed', ''),
                'ic': results.get('alpha_metrics', {}).get('ic', ''),
                'rank_ic': results.get('alpha_metrics', {}).get('rank_ic', ''),
                'sharpe_ratio': results.get('backtest_results', {}).get('sharpe_ratio', ''),
                'total_return': results.get('backtest_results', {}).get('total_return', ''),
                'max_drawdown': results.get('backtest_results', {}).get('max_drawdown', ''),
                'win_rate': results.get('backtest_results', {}).get('win_rate', ''),
                'n_trades': results.get('backtest_results', {}).get('n_trades', ''),
                'notes': notes or ''
            }
            
            writer.writerow(row)
    
    def load_experiments(self) -> List[Dict[str, Any]]:
        """
        Load all experiments from CSV.
        
        Returns:
            List of experiment dictionaries
        """
        if not self.log_file.exists():
            return []
        
        experiments = []
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiments.append(row)
        
        return experiments
    
    def get_best_experiment(self, metric: str = 'sharpe_ratio') -> Optional[Dict[str, Any]]:
        """
        Get best experiment by metric.
        
        Args:
            metric: Metric to optimize (default: 'sharpe_ratio')
        
        Returns:
            Best experiment dictionary or None
        """
        experiments = self.load_experiments()
        
        if not experiments:
            return None
        
        # Filter valid values
        valid_exps = []
        for exp in experiments:
            try:
                value = float(exp.get(metric, 0))
                if value != 0:
                    valid_exps.append((exp, value))
            except (ValueError, TypeError):
                continue
        
        if not valid_exps:
            return None
        
        # Sort by metric (descending)
        valid_exps.sort(key=lambda x: x[1], reverse=True)
        return valid_exps[0][0]


class JSONExperimentLogger:
    """
    JSON-based experiment logger.
    
    Logs experiments to JSON files for detailed tracking.
    """
    
    def __init__(self, log_dir: str = "experiments"):
        """
        Initialize JSON logger.
        
        Args:
            log_dir: Directory to store JSON logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log experiment to JSON.
        
        Args:
            experiment_id: Unique experiment identifier
            config: Experiment configuration
            results: Experiment results
            metadata: Optional metadata
        """
        log_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results,
            'metadata': metadata or {}
        }
        
        log_file = self.log_dir / f"{experiment_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
    
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load experiment from JSON.
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Experiment data or None
        """
        log_file = self.log_dir / f"{experiment_id}.json"
        
        if not log_file.exists():
            return None
        
        with open(log_file, 'r') as f:
            return json.load(f)

