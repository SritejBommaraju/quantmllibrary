#!/usr/bin/env python
"""
Profile entire pipeline to identify bottlenecks.

Usage:
    python scripts/profile_pipeline.py --config configs/base.yaml
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantml.config import load_config
from quantml.utils.profiling import PipelineProfiler, get_memory_usage
from quantml.data import load_csv_data
from quantml.training import FeaturePipeline
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training import QuantTrainer
from quantml.training.losses import mse_loss
from quantml import Tensor


def profile_pipeline(config_path: str):
    """Profile the entire pipeline."""
    print("Loading configuration...")
    config = load_config(config_path)
    
    profiler = PipelineProfiler()
    
    # Stage 1: Data Loading
    print("\n1. Profiling data loading...")
    profiler.start_stage('data_loading')
    data = load_csv_data(
        config.data.data_path,
        price_column='close',
        volume_column='volume'
    )
    profiler.end_stage('data_loading')
    
    # Stage 2: Feature Generation
    print("2. Profiling feature generation...")
    profiler.start_stage('feature_generation')
    pipeline = FeaturePipeline()
    pipeline.add_lagged_feature('price', lags=[1, 5, 10, 20])
    pipeline.add_rolling_feature('price', window=20, func='mean')
    features = pipeline.transform({'price': data['prices']})
    profiler.end_stage('feature_generation')
    
    # Stage 3: Model Creation
    print("3. Profiling model creation...")
    profiler.start_stage('model_creation')
    model = Linear(in_features=len(features[0]), out_features=1)
    optimizer = Adam(model.parameters(), lr=0.001)
    trainer = QuantTrainer(model, optimizer, mse_loss)
    profiler.end_stage('model_creation')
    
    # Stage 4: Training
    print("4. Profiling training...")
    profiler.start_stage('training')
    n_samples = min(100, len(features))  # Limit for profiling
    for i in range(n_samples):
        x = Tensor([features[i]])
        y = Tensor([[0.0]])  # Dummy target
        trainer.train_step(x, y)
    profiler.end_stage('training')
    
    # Stage 5: Inference
    print("5. Profiling inference...")
    profiler.start_stage('inference')
    for i in range(n_samples):
        x = Tensor([features[i]])
        _ = model.forward(x)
    profiler.end_stage('inference')
    
    # Print report
    profiler.print_report()
    
    # Memory usage
    final_memory = get_memory_usage()
    print(f"\nFinal Memory Usage:")
    print(f"  RSS: {final_memory['rss_mb']:.2f} MB")
    print(f"  VMS: {final_memory['vms_mb']:.2f} MB")
    if final_memory['percent'] > 0:
        print(f"  Percent: {final_memory['percent']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile QuantML pipeline')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    
    args = parser.parse_args()
    profile_pipeline(args.config)

