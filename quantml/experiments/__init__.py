"""
Experiment tracking module.

Provides tools for tracking experiments, comparing results, and ensuring reproducibility.
"""

from quantml.experiments.tracker import ExperimentTracker, compare_experiments
from quantml.experiments.logger import CSVExperimentLogger, JSONExperimentLogger
from quantml.experiments.results import (
    compare_experiments as compare_results,
    generate_summary_table,
    export_for_paper
)

__all__ = [
    'ExperimentTracker',
    'compare_experiments',
    'CSVExperimentLogger',
    'JSONExperimentLogger',
    'compare_results',
    'generate_summary_table',
    'export_for_paper'
]
