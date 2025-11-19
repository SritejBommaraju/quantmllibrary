"""
Centralized logging system for QuantML.

Provides structured logging with file rotation and experiment tracking.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from datetime import datetime
import json


def setup_logger(
    name: str = "quantml",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_file: Log file name (default: quantml_YYYYMMDD.log)
        console_output: Whether to output to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_dir is None:
        log_dir = "./logs"
    
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        log_file = f"quantml_{datetime.now().strftime('%Y%m%d')}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_experiment_start(
    logger: logging.Logger,
    config: Dict[str, Any],
    experiment_id: Optional[str] = None
):
    """
    Log experiment start with metadata.
    
    Args:
        logger: Logger instance
        config: Experiment configuration dictionary
        experiment_id: Optional experiment ID
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT START")
    logger.info("=" * 70)
    
    if experiment_id:
        logger.info(f"Experiment ID: {experiment_id}")
    
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Try to get git hash if available
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        logger.info(f"Git commit: {git_hash}")
    except Exception:
        pass
    
    logger.info("=" * 70)


def log_experiment_end(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    experiment_id: Optional[str] = None
):
    """
    Log experiment end with results.
    
    Args:
        logger: Logger instance
        metrics: Experiment metrics dictionary
        experiment_id: Optional experiment ID
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT END")
    logger.info("=" * 70)
    
    if experiment_id:
        logger.info(f"Experiment ID: {experiment_id}")
    
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    logger.info("=" * 70)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Log training progress.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss (optional)
        metrics: Additional metrics (optional)
    """
    msg = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
    if val_loss is not None:
        msg += f", Val Loss = {val_loss:.6f}"
    if metrics:
        metric_str = ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
        msg += f", {metric_str}"
    logger.info(msg)


def get_logger(name: str = "quantml") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

