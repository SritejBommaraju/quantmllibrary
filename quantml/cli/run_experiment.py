"""
Main CLI entry point for running QuantML experiments.

Usage:
    python -m quantml.cli.run_experiment --config configs/base.yaml
    python -m quantml.cli.run_experiment --instrument ES --start-date 2020-01-01 --end-date 2024-01-01
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantml.config import load_config, create_argparser, merge_config_with_args, ExperimentConfig
from quantml.utils.logging import setup_logger, log_experiment_start, log_experiment_end
from quantml.utils.reproducibility import set_random_seed, get_environment_info, create_experiment_id
from quantml.data import load_csv_data, validate_price_data, DataLoader
from quantml.data.feature_store import FeatureStore
from quantml.training import (
    QuantTrainer, FeaturePipeline, WalkForwardOptimizer, WindowType,
    AlphaEvaluator, BacktestEngine
)
from quantml.training.features import normalize_features
from quantml.models import Linear, SimpleRNN, TCN
from quantml.optim import SGD, Adam, RMSProp
from quantml.training.losses import mse_loss, sharpe_loss
from quantml import Tensor


def create_model(config: ExperimentConfig, n_features: int):
    """Create model from config."""
    model_type = config.model.model_type
    
    if model_type == "Linear":
        return Linear(
            in_features=n_features,
            out_features=config.model.out_features,
            bias=config.model.bias
        )
    elif model_type == "SimpleRNN":
        hidden_size = config.model.hidden_size or 32
        return SimpleRNN(
            input_size=n_features,
            hidden_size=hidden_size
        )
    elif model_type == "TCN":
        # Simplified TCN creation
        return Linear(
            in_features=n_features,
            out_features=config.model.out_features,
            bias=config.model.bias
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_optimizer(config: ExperimentConfig, model):
    """Create optimizer from config."""
    optimizer_type = config.training.optimizer
    lr = config.training.learning_rate
    
    params = model.parameters()
    
    if optimizer_type == "SGD":
        from quantml.optim import SGD
        return SGD(params, lr=lr)
    elif optimizer_type == "Adam":
        from quantml.optim import Adam
        return Adam(params, lr=lr)
    elif optimizer_type == "RMSProp":
        from quantml.optim import RMSProp
        return RMSProp(params, lr=lr)
    else:
        # Default to Adam
        from quantml.optim import Adam
        return Adam(params, lr=lr)


def create_loss_fn(config: ExperimentConfig):
    """Create loss function from config."""
    loss_name = config.training.loss_function
    
    if loss_name == "mse_loss":
        from quantml.training.losses import mse_loss
        return mse_loss
    elif loss_name == "sharpe_loss":
        from quantml.training.losses import sharpe_loss
        return sharpe_loss
    else:
        from quantml.training.losses import mse_loss
        return mse_loss


def run_experiment(config: ExperimentConfig, logger=None):
    """
    Run complete experiment pipeline.
    
    Args:
        config: Experiment configuration
        logger: Optional logger instance
    
    Returns:
        Dictionary with experiment results
    """
    if logger is None:
        logger = setup_logger(
            name="quantml",
            log_level=config.log_level,
            log_dir=os.path.join(config.output_dir, "logs")
        )
    
    experiment_id = create_experiment_id()
    logger.info(f"Starting experiment: {experiment_id}")
    
    # Set random seed for reproducibility
    set_random_seed(config.random_seed)
    
    # Log experiment start
    log_experiment_start(logger, config.to_dict(), experiment_id)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save environment info
    env_info = get_environment_info()
    env_info['experiment_id'] = experiment_id
    env_info['random_seed'] = config.random_seed
    
    import json
    with open(os.path.join(config.output_dir, 'environment.json'), 'w') as f:
        json.dump(env_info, f, indent=2)
    
    try:
        # Step 1: Load data
        logger.info("Loading data...")
        if config.data.data_source == "csv" and config.data.data_path:
            data = load_csv_data(
                config.data.data_path,
                price_column='close',
                volume_column='volume'
            )
        else:
            raise ValueError(f"Data source not supported: {config.data.data_source}")
        
        prices = data['prices']
        volumes = data.get('volumes', [100.0] * len(prices))
        
        logger.info(f"Loaded {len(prices)} data points")
        
        # Step 2: Validate data
        if config.data.validate_data:
            logger.info("Validating data...")
            is_valid, errors = validate_price_data(prices, volumes)
            if not is_valid:
                logger.warning(f"Data validation issues: {errors}")
        
        # Step 3: Create features
        logger.info("Creating features...")
        
        # Check feature cache
        feature_store = FeatureStore(cache_dir=config.data.feature_cache_path)
        cache_key = feature_store._generate_cache_key(
            config.data.instrument,
            config.data.start_date,
            config.data.end_date,
            {'features': config.features.enabled_features}
        )
        
        if config.data.cache_features and feature_store.cache_exists(cache_key):
            logger.info("Loading features from cache...")
            features, _ = feature_store.load_features(cache_key)
        else:
            # Create feature pipeline
            pipeline = FeaturePipeline()
            
            # Add configured features
            if 'lagged_price' in config.features.enabled_features:
                pipeline.add_lagged_feature('price', lags=config.features.lag_periods)
            
            if 'rolling_mean' in config.features.enabled_features:
                for window in config.features.rolling_windows:
                    pipeline.add_rolling_feature('price', window=window, func='mean')
            
            if 'rolling_std' in config.features.enabled_features:
                for window in config.features.rolling_windows:
                    pipeline.add_rolling_feature('price', window=window, func='std')
            
            if 'returns' in config.features.enabled_features:
                pipeline.add_time_series_feature('price', 'returns')
            
            if 'volatility' in config.features.enabled_features:
                pipeline.add_time_series_feature('price', 'volatility', n=20)
            
            # Transform
            features = pipeline.transform({'price': prices})
            
            # Normalize
            if config.features.normalize:
                features = normalize_features(
                    features,
                    method=config.features.normalization_method
                )
            
            # Cache features
            if config.data.cache_features:
                feature_store.save_features(
                    features,
                    config.data.instrument,
                    config.data.start_date,
                    config.data.end_date,
                    {'features': config.features.enabled_features}
                )
        
        logger.info(f"Created {len(features)} samples with {len(features[0])} features")
        
        # Step 4: Create targets
        targets = []
        for i in range(len(prices) - 1):
            ret = (prices[i + 1] - prices[i]) / prices[i] if prices[i] > 0 else 0.0
            targets.append(ret)
        
        features = features[:-1]  # Align
        
        # Step 5: Train model
        logger.info("Training model...")
        
        model = create_model(config, len(features[0]))
        optimizer = create_optimizer(config, model)
        loss_fn = create_loss_fn(config)
        
        trainer = QuantTrainer(model, optimizer, loss_fn)
        
        # Walk-forward training if enabled
        if config.training.walk_forward['enabled']:
            wfo = WalkForwardOptimizer(
                window_type=WindowType.EXPANDING if config.training.walk_forward['window_type'] == 'expanding' else WindowType.ROLLING,
                train_size=config.training.walk_forward['train_size'],
                test_size=config.training.walk_forward['test_size']
            )
            
            all_predictions = []
            all_actuals = []
            
            for train_idx, test_idx in wfo.split(features, n_splits=3):
                # Train
                X_train = [features[i] for i in train_idx]
                y_train = [targets[i] for i in train_idx]
                
                for epoch in range(config.training.epochs):
                    for i in range(0, len(X_train), config.training.batch_size):
                        batch_x = X_train[i:i+config.training.batch_size]
                        batch_y = y_train[i:i+config.training.batch_size]
                        
                        for x, y in zip(batch_x, batch_y):
                            x_tensor = Tensor([x])
                            y_tensor = Tensor([[y]])
                            trainer.train_step(x_tensor, y_tensor)
                
                # Test
                X_test = [features[i] for i in test_idx]
                y_test = [targets[i] for i in test_idx]
                
                for i in range(len(X_test)):
                    x = Tensor([X_test[i]])
                    pred = model.forward(x)
                    pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
                    all_predictions.append(pred_val)
                    all_actuals.append(y_test[i])
            
            predictions = all_predictions
            actuals = all_actuals
        else:
            # Simple train/test split
            train_size = int(len(features) * 0.7)
            X_train = features[:train_size]
            y_train = targets[:train_size]
            X_test = features[train_size:]
            y_test = targets[train_size:]
            
            # Train
            for epoch in range(config.training.epochs):
                for i in range(0, len(X_train), config.training.batch_size):
                    batch_x = X_train[i:i+config.training.batch_size]
                    batch_y = y_train[i:i+config.training.batch_size]
                    
                    for x, y in zip(batch_x, batch_y):
                        x_tensor = Tensor([x])
                        y_tensor = Tensor([[y]])
                        trainer.train_step(x_tensor, y_tensor)
            
            # Test
            predictions = []
            actuals = []
            for i in range(len(X_test)):
                x = Tensor([X_test[i]])
                pred = model.forward(x)
                pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
                predictions.append(pred_val)
                actuals.append(y_test[i])
        
        # Step 6: Evaluate
        logger.info("Evaluating results...")
        
        evaluator = AlphaEvaluator(predictions, actuals)
        alpha_metrics = evaluator.evaluate()
        
        # Step 7: Backtest
        test_prices = prices[-len(predictions):] if len(predictions) < len(prices) else prices[len(prices)-len(predictions):]
        backtest = BacktestEngine(initial_capital=100000.0)
        backtest_results = backtest.run_with_predictions(predictions, test_prices, targets=actuals)
        
        # Combine results
        results = {
            'experiment_id': experiment_id,
            'alpha_metrics': alpha_metrics,
            'backtest_results': backtest_results,
            'config': config.to_dict()
        }
        
        # Save results
        results_path = os.path.join(config.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Log experiment end
        log_experiment_end(logger, {
            'ic': alpha_metrics['ic'],
            'sharpe': backtest_results['sharpe_ratio'],
            'return': backtest_results['total_return']
        }, experiment_id)
        
        return results
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


def main():
    """Main CLI entry point."""
    parser = create_argparser()
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Create default config
        config = ExperimentConfig()
    
    # Merge CLI arguments
    config = merge_config_with_args(config, args)
    
    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Run experiment
    try:
        results = run_experiment(config)
        print("\n" + "=" * 70)
        print("Experiment completed successfully!")
        print("=" * 70)
        print(f"Experiment ID: {results['experiment_id']}")
        print(f"IC: {results['alpha_metrics']['ic']:.4f}")
        print(f"Sharpe Ratio: {results['backtest_results']['sharpe_ratio']:.4f}")
        print(f"Total Return: {results['backtest_results']['total_return']*100:.2f}%")
        print(f"Results saved to: {config.output_dir}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

