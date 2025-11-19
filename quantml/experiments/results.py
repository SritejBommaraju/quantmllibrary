"""
Experiment results comparison and analysis.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path


def compare_experiments(experiment_dirs: List[str]) -> Dict[str, Any]:
    """
    Compare multiple experiment runs.
    
    Args:
        experiment_dirs: List of experiment directory paths
    
    Returns:
        Comparison dictionary
    """
    experiments = []
    
    for exp_dir in experiment_dirs:
        results_path = Path(exp_dir) / 'results.json'
        metadata_path = Path(exp_dir) / 'metadata.json'
        
        results = {}
        metadata = {}
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        experiments.append({
            'dir': str(exp_dir),
            'metadata': metadata,
            'results': results
        })
    
    # Extract key metrics
    comparison = {
        'n_experiments': len(experiments),
        'experiments': []
    }
    
    for exp in experiments:
        alpha_metrics = exp['results'].get('alpha_metrics', {})
        backtest_results = exp['results'].get('backtest_results', {})
        
        comparison['experiments'].append({
            'experiment_id': Path(exp['dir']).name,
            'model_type': exp['metadata'].get('model_type', ''),
            'optimizer': exp['metadata'].get('optimizer', ''),
            'learning_rate': exp['metadata'].get('learning_rate', ''),
            'ic': alpha_metrics.get('ic', 0.0),
            'rank_ic': alpha_metrics.get('rank_ic', 0.0),
            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0.0),
            'total_return': backtest_results.get('total_return', 0.0),
            'max_drawdown': backtest_results.get('max_drawdown', 0.0),
            'win_rate': backtest_results.get('win_rate', 0.0)
        })
    
    return comparison


def generate_summary_table(comparison: Dict[str, Any]) -> str:
    """
    Generate summary table for paper.
    
    Args:
        comparison: Comparison dictionary
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("Experiment Comparison")
    lines.append("=" * 100)
    lines.append(f"{'Experiment':<20} {'Model':<10} {'IC':<8} {'Sharpe':<8} {'Return':<10} {'MaxDD':<8}")
    lines.append("-" * 100)
    
    for exp in comparison['experiments']:
        lines.append(
            f"{exp['experiment_id']:<20} "
            f"{exp['model_type']:<10} "
            f"{exp['ic']:>7.4f} "
            f"{exp['sharpe_ratio']:>7.4f} "
            f"{exp['total_return']*100:>9.2f}% "
            f"{exp['max_drawdown']:>7.4f}"
        )
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


def export_for_paper(comparison: Dict[str, Any], output_path: str, format: str = 'json'):
    """
    Export results for paper.
    
    Args:
        comparison: Comparison dictionary
        output_path: Output file path
        format: Export format ('json', 'csv', 'latex')
    """
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
    
    elif format == 'csv':
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id', 'model_type', 'optimizer', 'ic', 'sharpe_ratio',
                'total_return', 'max_drawdown'
            ])
            writer.writeheader()
            for exp in comparison['experiments']:
                writer.writerow({
                    'experiment_id': exp['experiment_id'],
                    'model_type': exp['model_type'],
                    'optimizer': exp['optimizer'],
                    'ic': exp['ic'],
                    'sharpe_ratio': exp['sharpe_ratio'],
                    'total_return': exp['total_return'],
                    'max_drawdown': exp['max_drawdown']
                })
    
    elif format == 'latex':
        # Generate LaTeX table
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\hline")
        lines.append("Experiment & Model & IC & Sharpe & Return \\\\")
        lines.append("\\hline")
        
        for exp in comparison['experiments']:
            lines.append(
                f"{exp['experiment_id']} & "
                f"{exp['model_type']} & "
                f"{exp['ic']:.4f} & "
                f"{exp['sharpe_ratio']:.4f} & "
                f"{exp['total_return']*100:.2f}\\% \\\\"
            )
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Experiment Results}")
        lines.append("\\end{table}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

