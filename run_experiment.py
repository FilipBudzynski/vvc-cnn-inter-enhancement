#!/usr/bin/env python3
"""
Training script for VVC enhancement experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhancer.models.discriminator import Discriminator
from enhancer.models.vtm_enhancer import Enhancer
from enhancer.yuv_dataset import YUVVTMDataset, create_yuv_vtm_dataloader
from enhancer.trainer_module import TrainerModule
from enhancer.utils import weights_init
from enhancer.config import Config, TrainingMode
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="train",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="test",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="predict",
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        default="configs/experiments/enhancer_dense.yaml",
        help="config file",
    )

    args = parser.parse_args()

    config = Config.load(args.config)

    # Create YUV+VTM dataloaders using config
    train_loader, val_loader = create_yuv_vtm_dataloader(
        data_dir="output",  # Directory with YUV and CSV files  
        batch_size=config.dataloader.batch_size,
        chunk_size=(config.dataset.train.chunk_height, config.dataset.train.chunk_width)
    )
    
    print("✅ YUV+VTM DataModule created successfully!")

    enhancer = Enhancer(
        config=config.enhancer,
        use_vtm_features=True,
        vtm_feature_channels=10,
    )

    if config.enhancer.load_from:
        enhancer.load_state_dict(torch.load(config.enhancer.load_from), strict=True)
        print("loaded enhancer")
    else:
        enhancer.apply(weights_init)

    discriminator = Discriminator(
        config=config.discriminator,
    )

    if config.discriminator.load_from:
        discriminator.load_state_dict(
            torch.load(config.discriminator.load_from), strict=True
        )
        print("loaded discriminator")
    else:
        discriminator.apply(weights_init)

    module = TrainerModule(
        config.trainer,
        enhancer,
        discriminator,
        test_full_frames=config.test_full_frames,
    )

    wandb_logger = WandbLogger(
        project="vvc-enhancer",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=config.trainer.current.epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath="checkpoints", filename="{epoch}"),
        ],
        logger=wandb_logger,
    )

    if args.train:
        trainer.fit(module, train_loader)

    if config.enhancer.save_to:
        torch.save(enhancer.state_dict(), config.enhancer.save_to)

    if config.discriminator.save_to:
        torch.save(discriminator.state_dict(), config.discriminator.save_to)

    if args.test:
        trainer.test(module, val_loader)

    if args.predict:
        trainer.predict(module, val_loader)