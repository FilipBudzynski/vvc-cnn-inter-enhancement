import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from enhancer.datamodule import VVCDataModule
from enhancer.trainer import TrainerModule
from enhancer.models.enhancer import Enhancer
from enhancer.config import Config, NetworkImplementation

from enhancer.config import TrainerConfig, ModeTrainingConfig, TrainingMode

torch.serialization.add_safe_globals([TrainerConfig, ModeTrainingConfig, TrainingMode])


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--checkpoint", "-ckpt", type=str, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)

    data_module = VVCDataModule(
        dataset_config=config.dataset, dataloader_config=config.dataloader
    )

    model = Enhancer(config.enhancer)

    module = TrainerModule(config=config.trainer, enhancer=model)

    run_name = "Lanister-resnet-11ch-32QP"
    wandb_logger = WandbLogger(project="vvc-cnn-inter", name=run_name)

    trainer = pl.Trainer(
        num_sanity_val_steps=1,
        accelerator="gpu",
        devices=1,
        max_epochs=config.trainer.current.epochs,
        precision="16-mixed",
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename=run_name + "-{epoch:02d}-{val_psnr_Y:.2f}",
                monitor="val_psnr_Y",
                mode="max",
                save_top_k=2,
            ),
        ],
        logger=wandb_logger,
    )

    if args.train:
        trainer.fit(module, data_module)
        if config.enhancer.save_to:
            torch.save(model.state_dict(), config.enhancer.save_to)

    if args.test:
        trainer.test(module, data_module, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
