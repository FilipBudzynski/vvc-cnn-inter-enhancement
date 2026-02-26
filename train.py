"""
Training Script
================

Main training entry point with configurable model and training parameters

Usage:
    python train.py --config configs/experiments/dense_v1.yaml
    python train.py --model dense --epochs 50 --batch_size 4
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import time

from configs import (
    NetworkImplementation,
    ModelConfig, 
    TrainingConfig, 
    DataConfig,
    load_model_config,
    load_training_config,
    load_data_config
)
from models.vtm_enhancer import VTMReferenceEnhancer
from data.dataset import VTMDataset
from training.trainer import VTMTrainer


def create_model_from_config(model_config: ModelConfig) -> nn.Module:
    """Create model from configuration"""
    if model_config.implementation.value == "dense":
        config = model_config.dense_config
        return VTMReferenceEnhancer(
            input_channels=config.input_channels,
            metadata_channels=config.metadata_encoder.metadata_channels,
            metadata_features=config.metadata_encoder.metadata_features
        )
    else:
        raise ValueError(f"Model {model_config.implementation} not yet implemented")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train VTM-enhanced video enhancer")
    
    # Configuration file option
    parser.add_argument("--config", type=str, help="YAML configuration file")
    
    # Model options
    parser.add_argument("--model", type=str, choices=["dense", "res", "conv"], 
                       help="Model architecture")
    parser.add_argument("--growth_rate", type=int, help="DenseNet growth rate")
    parser.add_argument("--metadata_features", type=int, help="Metadata encoder features")
    
    # Training options
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "adamw"], help="Optimizer")
    
    # Data options
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--target_size", type=int, nargs=2, help="Target image size (height width)")
    
    # Experiment options
    parser.add_argument("--experiment", type=str, default="default", help="Experiment name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    
    return parser.parse_args()


def update_configs_from_args(args):
    """Update configs based on command line arguments"""
    model_config = load_model_config(args.config) if args.config else load_model_config()
    training_config = load_training_config(args.config) if args.config else load_training_config()
    data_config = load_data_config(args.config) if args.config else load_data_config()
    
    # Override with command line arguments 
    if args.model:
        from configs.model_config import NetworkImplementation
        if model_config is None: 
            if args.model == "dense":
                model_config = load_model_config()

        model_config.implementation = NetworkImplementation(args.model)
    
    # Override with command line arguments using factory setter methods
    if args.growth_rate and model_config and model_config.dense_config:
        model_config.dense_config.growth_rate = args.growth_rate 
    
    if args.metadata_features and model_config and model_config.dense_config:
        model_config.dense_config.metadata_encoder.metadata_features = args.metadata_features
    
    if args.epochs and training_config:
        training_config.epochs = args.epochs
    
    if args.batch_size and training_config:
        training_config.batch_size = args.batch_size
    
    if args.learning_rate and training_config:
        training_config.learning_rate = args.learning_rate
    
    if args.optimizer and training_config:
        from configs.training_config import OptimizerType
        training_config.optimizer = OptimizerType(args.optimizer)
    
    if args.data_dir and data_config:
        data_config.data_dir = args.data_dir
    
    if args.target_size and data_config:
        data_config.target_size = tuple(args.target_size)
    
    if args.experiment:
        if training_config:
            training_config.experiment_name = args.experiment
    
    return model_config, training_config, data_config


def save_experiment_config(model_config, training_config, data_config, save_dir):
    """Save experiment configuration"""
    config_dict = {
        "model": model_config.model_dump() if model_config else None,
        "training": training_config.model_dump() if training_config else None,
        "data": data_config.model_dump() if data_config else None
    }
    
    save_path = Path(save_dir) / "experiment_config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Saved experiment config to {save_path}")


def main():
    """Main training function"""
    args = parse_args()
    model_config, training_config, data_config = update_configs_from_args(args)
    
    # Set device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Create experiment directory
    experiment_dir = Path("results") / "experiments" / training_config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    save_experiment_config(model_config, training_config, data_config, experiment_dir)
    
    # Create model
    model = create_model_from_config(model_config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_config.implementation.value if model_config else 'default'}")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Create trainer
    trainer = VTMTrainer(
        model=model,
        config=training_config,
        data_config=data_config,
        device=device,
        save_dir=experiment_dir
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
