from models.discriminator import Discriminator
from models.enhancer import Enhancer
from datamodule import VVCDataModule
from trainer_module import TrainerModule
from utils import weights_init
from config import Config, TrainingMode
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
        help="train",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="train",
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        default="config.yaml",
        help="config file",
    )

    args = parser.parse_args()

    config = Config.load(args.config)

    data_module = VVCDataModule(
        dataset_config=config.dataset,
        dataloader_config=config.dataloader,
        test_full_frames=config.test_full_frames,
    )

    enhancer = Enhancer(
        config=config.enhancer,
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
        trainer.fit(module, data_module)

    if config.enhancer.save_to:
        torch.save(enhancer.state_dict(), config.enhancer.save_to)

    if config.discriminator.save_to:
        torch.save(discriminator.state_dict(), config.discriminator.save_to)

    if args.test:
        trainer.test(module, data_module)

    if args.predict:
        trainer.predict(module, data_module)
