import os
import glob
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import argparse
import time
from im2mesh import config, data

from collections import OrderedDict

from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch3d
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)

# Arguments
parser = argparse.ArgumentParser(
    description='Training function.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for train and val loaders.')
parser.add_argument('--epochs-per-run', type=int, default=-1,
                    help='Number of epochs to train before restart.')
parser.add_argument('--run-name', type=str, default='',
                    help='Run name for Wandb logging.')

if  __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    num_workers = args.num_workers
    epochs_per_run = args.epochs_per_run

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    gpus = cfg['training']['gpus']

    checkpoint_every = cfg['training']['checkpoint_every_n_epochs']
    validate_every = cfg['training']['validate_every_n_epochs']
    max_epochs = cfg['training']['max_epochs']

    exit_after = args.exit_after

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Dataloaders
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)

    # Here batch_size is batch_size per GPU
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=args.num_workers, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False
    )

    # Create PyTorch Lightning model
    model = config.get_model(cfg, dataset=train_dataset, val_size=len(val_loader))

    # Create logger
    latest_wandb_path = glob.glob(os.path.join(out_dir, 'wandb', 'latest-run', 'run-*.wandb'))
    if len(latest_wandb_path) == 1:
        run_id = os.path.basename(latest_wandb_path[0]).split('.')[0][4:]
    else:
        run_id = None

    if len(args.run_name) > 0:
        run_name = args.run_name
    else:
        run_name = None

    kwargs = {'settings': wandb.Settings(start_method='fork')}
    logger = pl.loggers.WandbLogger(name=run_name,
                                    project='arah',
                                    id=run_id,
                                    save_dir=out_dir,
                                    config=cfg,
                                    **kwargs)

    # Create PyTorch Lightning trainer
    checkpoint_callback = ModelCheckpoint(save_top_k=0,
                                          dirpath=os.path.join(out_dir, 'checkpoints'),
                                          every_n_epochs=checkpoint_every,
                                          save_on_train_epoch_end=True,
                                          save_last=True)

    checkpoint_path = os.path.join(out_dir, 'checkpoints/last.ckpt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = None

    if epochs_per_run <= 0:
        # epochs_per_run is not specified: we train with max_epochs and validate
        # this usually applies for training on local machines
        pass
    else:
        # epochs_per_run is specified: we train with already trained epochs + epochs_per_run,
        # and do not validate
        # this usually applies for training on HPC cluster with jon-chaining
        if checkpoint_path is None:
            max_epochs = epochs_per_run
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            max_epochs = min(ckpt['epoch'] + epochs_per_run, max_epochs)
            del ckpt

        validate_every = max_epochs + 1

    trainer = pl.Trainer(logger=logger,
                         log_every_n_steps=10,
                         default_root_dir=out_dir,
                         callbacks=[checkpoint_callback],
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=validate_every,
                         accelerator='gpu',
                         strategy='ddp' if len(gpus) > 1 else None,
                         devices=gpus,
                         num_sanity_val_steps=0)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=checkpoint_path)
