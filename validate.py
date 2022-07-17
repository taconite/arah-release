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
    description='Validation function on with-distribution poses (ZJU training and testing).'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--novel-pose', action='store_true', help='Test on novel-poses.')
parser.add_argument('--novel-pose-view', type=str, default=None, help='Novel view to use for rendering novel poses. Specify this argument if you only want to render a specific view of novel poses.')
parser.add_argument('--novel-view', action='store_true', help='Test on novel-views of all training poses.')
parser.add_argument('--multi-gpu', action='store_true', help='Test on multiple GPUs.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for val/test loaders.')
parser.add_argument('--run-name', type=str, default='',
                    help='Run name for Wandb logging.')

if  __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    num_workers = args.num_workers

    # Novel-view synthesis on training poses: evluate every 30th frame
    if args.novel_view and not args.novel_pose:
        cfg['data']['val_subsampling_rate'] = 30

    # View-synthesis (can be either training or testing views) on novel poses
    if args.novel_pose_view is not None:
        assert (args.novel_pose)
        cfg['data']['test_subsampling_rate'] = 1
        cfg['data']['test_views'] = [args.novel_pose_view]

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']

    # Dataloaders
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('test' if args.novel_pose else 'val', cfg)

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
    checkpoint_path = os.path.join(out_dir, 'checkpoints/last.ckpt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')

    if args.multi_gpu:
        trainer = pl.Trainer(logger=logger,
                             default_root_dir=out_dir,
                             accelerator='gpu',
                             strategy='ddp',
                             devices=[0, 1, 2, 3],
                             num_sanity_val_steps=0)
    else:
        trainer = pl.Trainer(logger=logger,
                             default_root_dir=out_dir,
                             accelerator='gpu',
                             devices=[0],
                             num_sanity_val_steps=0)

    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=checkpoint_path, verbose=True)
