import time
from argparse import ArgumentParser
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.privacy_network import configs
from src.privacy_network.datasets import PhotoDataModule
from src.privacy_network.model import Model


def main(args):
    seed_everything(args.seed, workers=True)

    config = configs.hyper_params | getattr(configs, args.config)  
    run_dir = Path("output") / config["name"] / "privacy_network" / f"{args.config}_seed{args.seed}"
    if (run_dir / "best.ckpt").exists() and not args.fast_dev_run:
        raise SystemExit(f"{run_dir} already has a checkpoint, delete the folder to rerun")
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        accelerator = "gpu"
        precision = "bf16-mixed"
    else:
        accelerator = "cpu"
        precision = "32"

    model = Model(config, args.epochs)
    data = PhotoDataModule(config, model.model.encoder.transform)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        limit_train_batches=config.get("limit_train_batches", 1.0),
        accelerator=accelerator,
        precision=precision,
        callbacks=[
            ModelCheckpoint(
                dirpath=run_dir,
                filename="best",
                monitor="val/loss",
                mode="min",
                save_last=True,
                enable_version_counter=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        # version="" puts the logs in tb/ and csv/, no version_0, version_1, ...
        logger=[
            TensorBoardLogger(run_dir, name="tb", version=""),
            CSVLogger(run_dir, name="csv", version=""),
        ],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    start = time.time()  # Train and track time
    trainer.fit(model, datamodule=data)
    if not args.fast_dev_run:
        (run_dir / "train_minutes.txt").write_text(f"{(time.time() - start) / 60:.1f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--config", default="osv5m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast_dev_run", action="store_true")
    main(parser.parse_args())