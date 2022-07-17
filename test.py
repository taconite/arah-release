import os
import glob
import torch
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
    description='Test function that renders images without quantitative evaluation.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--pose-dir', type=str, default='gBR_sBM_cAll_d04_mBR1_ch06_view1', help='Which out-of-distribution pose sequence to render.')
parser.add_argument('--test-views', type=str, default='1', help='Which views to render.')
parser.add_argument('--subsampling-rate', type=int, default=1, help='Sampling rate for poses. Larger rate means less poses to render.')
parser.add_argument('--start-frame', type=int, default=0, help='Frame index to start rendering.')
parser.add_argument('--end-frame', type=int, default=0, help='Frame index to stop rendering.')
parser.add_argument('--low-vram', action='store_true', help='Use less VRAM for inference.')
parser.add_argument('--multi-gpu', action='store_true', help='Test on multiple (4) GPUs.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for val/test loaders.')

if  __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    num_workers = args.num_workers

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']

    # Overwrite configuration
    cfg['data']['test_views'] = args.test_views.split(',')
    cfg['data']['dataset'] = 'zju_mocap_odp'
    cfg['data']['path'] = 'data/odp'
    cfg['data']['test_subsampling_rate'] = args.subsampling_rate
    cfg['data']['test_start_frame'] = args.start_frame
    cfg['data']['test_end_frame'] = args.end_frame
    cfg['data']['pose_dir'] = args.pose_dir
    val_dataset = config.get_dataset('test', cfg)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False
    )

    checkpoint_path = os.path.join(out_dir, 'checkpoints/last.ckpt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')

    # Create PyTorch Lightning model
    model = config.get_model(cfg, val_size=len(val_loader), mode='test', low_vram=args.low_vram, checkpoint_path=checkpoint_path)

    # Create PyTorch Lightning trainer
    if args.multi_gpu:
        trainer = pl.Trainer(default_root_dir=out_dir,
                             accelerator='gpu',
                             strategy='ddp',
                             devices=[0, 1, 2, 3],
                             num_sanity_val_steps=0)
    else:
        trainer = pl.Trainer(default_root_dir=out_dir,
                             accelerator='gpu',
                             devices=[0],
                             num_sanity_val_steps=0)

    trainer.test(model=model, dataloaders=val_loader, verbose=True)
