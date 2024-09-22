import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from dataclasses import dataclass
from typing import Tuple
from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.utils import load_metadata_hdf5
import wandb
from dataclass_dict_convert import dataclass_dict_convert 
from pytorch_lightning.loggers import WandbLogger
import hydra
from pathlib import Path

@hydra.main(config_name="config")
def main(cfg):
    seed_everything(9)

    # Get number of folds from the config
    num_folds = cfg.num_folds
    
    for fold in range(num_folds):
    print(f"Training fold {fold + 1}/{num_folds}")

    # Set up paths for the current fold
    train_path = Path(hydra.utils.to_absolute_path(f"{cfg.train_path}_fold{fold}"))
    val_path = Path(hydra.utils.to_absolute_path(f"{cfg.val_path}_fold{fold}"))

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Train or validation path for fold {fold} does not exist.")

    # Set up data for the fold
    data = DataModule(
        train_path,
        val_path,
        None,
        cfg
    )

    # Initialize the model
    model = Model(cfg=cfg.architecture)

    # Setup WandB logger if enabled
    if cfg.wandb:
        wandb.init(project="ICML", name=f"fold_{fold}")
        config = wandb.config
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None

    # Checkpoint callback for saving the best model for the current fold
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"Exp_weights/fold_{fold}/",
        filename=train_path.stem + f"_fold{fold}_log_" + "-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    # Define the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        distributed_backend="ddp",
        gpus=cfg.gpu,
        max_epochs=cfg.epochs,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Start training for the current fold
    trainer.fit(model, data)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Modify GPU as needed
    main()
