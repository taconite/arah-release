import os
import torch
import numpy as np
from torch import nn
from scipy.spatial.transform import Rotation

from im2mesh.metaavatar import models as metaavatar_models

from im2mesh.metaavatar_render import models
from im2mesh.metaavatar_render.models.decoder import RenderingNetwork, SingleVarianceNetwork
from im2mesh.metaavatar_render.models.skinning_model import SkinningModel

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

def get_sdf_decoder(cfg, init_weights=True):
    ''' Returns a SDF decoder instance.

    Args:
        cfg (yaml config): yaml config object
        init_weights (bool): whether to initialize the weights for the SDF network with pre-trained model (MetaAvatar)
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    assert (decoder is not None)

    decoder = metaavatar_models.decoder_dict[decoder](**decoder_kwargs)#.to(device)

    if init_weights:
        optim_geometry_net_path = cfg['model']['geometry_net']
        ckpt = torch.load(optim_geometry_net_path, map_location='cpu')

        decoder_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('decoder'):
                decoder_state_dict[k[8:]] = v

        decoder.load_state_dict(decoder_state_dict, strict=False)

    return decoder


def get_skinning_decoder(cfg, dim=3):
    ''' Returns skinning decoder instances, including forward and backward decoders.

    Args:
        cfg (yaml config): yaml config object
        dim (int): points dimension
    '''
    decoder = cfg['model']['skinning_decoder']
    decoder_kwargs = cfg['model']['skinning_decoder_kwargs']

    if decoder is not None:
        decoder = metaavatar_models.decoder_dict[decoder](**decoder_kwargs)
    else:
        decoder = None

    return decoder


def get_skinning_model(cfg, dim=3, init_weights=True):
    ''' Returns Skinning Model instances.

    Args:
        cfg (yaml config): yaml config object
        dim (int): points dimension
        init_weights (bool): whether to initialize the weights for the skinning network with pre-trained model (MetaAvatar)
    '''
    decoder = get_skinning_decoder(cfg, dim=dim)

    optim_skinning_net_path = cfg['model']['skinning_net2']
    if init_weights and optim_skinning_net_path is not None:
        ckpt = torch.load(optim_skinning_net_path, map_location='cpu')

        skinning_decoder_fwd_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('skinning_decoder_fwd'):
                skinning_decoder_fwd_state_dict[k[21:]] = v

        decoder.load_state_dict(skinning_decoder_fwd_state_dict, strict=False)

    skinning_model = SkinningModel(skinning_decoder_fwd=decoder)

    return skinning_model


def get_color_decoder(cfg):
    ''' Returns skinning encoder instances, including forward and backward encoders.

    Args:
        cfg (yaml config): yaml config object
    '''
    renderer = cfg['model']['renderer']
    pose_encoder = cfg['model']['color_pose_encoder']
    renderer_kwargs = cfg['model']['renderer_kwargs']
    latent_dim = cfg['model']['latent_dim']
    if renderer == 'mlp':
        if pose_encoder is None:
            # No pose feature for color network
            d_feature = 256
        elif pose_encoder == 'leap':
            # LEAP pose encoder as pose feature
            d_feature = 256 + 144
        elif pose_encoder == 'root':
            # Root transformation as pose feature
            d_feature = 256 + 12
        elif pose_encoder == 'latent':
            # Per-frame 128-dimentional latent code as pose feature
            d_feature = 256 + latent_dim
        elif pose_encoder == 'hybrid':
            # Root transformation plus per-frame 128-dimentional latent code as pose feature
            d_feature = 256 + 12 + latent_dim
        else:
            raise ValueError('Unsupported rendering network pose encoder {}. Supported encoders are: None, leap, root, latent and hybrid')

        rendering_network = RenderingNetwork(d_feature=d_feature,
                                             pose_encoder=pose_encoder,
                                             **renderer_kwargs)
    else:
        raise ValueError('Supported renderer types are: mlp. Got {}'.format(renderer))

    logger.info(rendering_network)

    return rendering_network


def get_deviation_decoder(cfg):
    ''' Returns learnable scaling parameter, wrapped as a nn.Module.

    Args:
        cfg (yaml config): yaml config object
    '''
    deviation_network = SingleVarianceNetwork(1e-3)

    return deviation_network


def get_model(cfg, **kwargs):
    ''' Return the model instance.

    Args:
        cfg (dict): imported yaml config
    '''

    init_weights = True

    mode = kwargs.get('mode', None)
    if mode is not None and mode in ['val', 'test']:
        init_weights = False

    sdf_decoder = get_sdf_decoder(cfg, init_weights=init_weights)
    skinning_model = get_skinning_model(cfg, init_weights=init_weights)
    color_decoder = get_color_decoder(cfg)
    deviation_decoder = get_deviation_decoder(cfg)

    # Get initial camera extrinsics, if we want to optimize them
    train_cameras = cfg['model']['train_cameras']
    if train_cameras:
        dataset = kwargs['dataset']
        cam_names = dataset.cam_names
        cameras = dataset.cameras
        cam_rots = [Rotation.from_matrix(cameras[cam_name]['R']).as_quat().astype(np.float32) for cam_name in cam_names]
        cam_trans = [np.array(cameras[cam_name]['T'], dtype=np.float32).ravel() for cam_name in cam_names]
        cam_rots = np.stack(cam_rots, axis=0)
        cam_trans = np.stack(cam_trans, axis=0)
        kwargs.update({'cam_rots': cam_rots, 'cam_trans': cam_trans})

    # Get initial SMPL poses/shapes, if we want to optimize them
    train_smpl = cfg['model']['train_smpl']
    if train_smpl:
        dataset = kwargs['dataset']
        root_orient = []
        pose_body = []
        pose_hand = []
        betas = []
        trans = []
        frames = []
        cam_idx = dataset.data[0]['cam_idx']
        for d_idx, single_data in enumerate(dataset.data):
            if single_data['cam_idx'] != cam_idx:
                # We assume one batch from dataloader consists of data from the same
                # frame, so we only keep SMPL parameters for one camera-view
                break

            data_path = single_data['model_file']
            model_dict = np.load(data_path)

            root = model_dict['root_orient'].astype(np.float32)
            if (root == 0.0).all():
                root += 1e-8

            root_orient.append(root)

            body = model_dict['pose_body'].astype(np.float32).reshape([-1, 3])
            body[(body == 0.0).all(axis=-1)] += 1e-8
            pose_body.append(body.reshape(-1))

            hand = model_dict['pose_hand'].astype(np.float32).reshape([-1, 3])
            hand[(hand == 0.0).all(axis=-1)] += 1e-8
            pose_hand.append(hand.reshape(-1))

            if d_idx == 0:
                # Since it is a single subject, we use a single beta for all frames
                betas = model_dict['betas'].astype(np.float32)
                gender = single_data['gender']

            trans.append(model_dict['trans'].astype(np.float32))
            frames.append(single_data['frame_idx'])

        kwargs.update({'root_orient': root_orient,
                       'pose_body': pose_body,
                       'pose_hand': pose_hand,
                       'trans': trans,
                       'betas': betas,
                       'gender': gender,
                       'frames': frames})

    # Get initial geometry/appearance latent codes, if applicable
    train_latent_code = cfg['model']['color_pose_encoder'] in ['hybrid', 'latent']
    train_geo_latent_code = cfg['model']['geo_pose_encoder'] in ['latent']
    if train_latent_code or train_geo_latent_code:
        dataset = kwargs['dataset']
        cam_idx = dataset.data[0]['cam_idx']
        n_data_points = 0
        frames = []
        for d_idx, single_data in enumerate(dataset.data):
            if single_data['cam_idx'] != cam_idx:
                # We assume one batch from dataloader consists of data from the same
                # frame, so we only keep latent code for one camera-view
                break

            n_data_points += 1
            frames.append(single_data['frame_idx'])

        kwargs.update({'n_data_points': n_data_points,
                       'frames': frames})

    cano_view_dirs = cfg['model']['cano_view_dirs']
    near_surface_samples = cfg['model']['near_surface_samples']
    far_surface_samples = cfg['model']['far_surface_samples']
    n_steps = cfg['model']['n_steps']
    train_skinning_net = cfg['training']['train_skinning_net']
    render_last_pt = cfg['model']['render_last_pt']
    pose_input_noise = cfg['training']['pose_input_noise']
    view_input_noise = cfg['training']['view_input_noise']
    nv_noise_type = cfg['training']['nv_noise_type']
    # Get full MetaAvatarRender model
    model = models.MetaAvatarRender(
        sdf_decoder=sdf_decoder,
        skinning_model=skinning_model,
        color_decoder=color_decoder,
        deviation_decoder=deviation_decoder,
        train_cameras=train_cameras,
        train_smpl=train_smpl,
        train_latent_code=train_latent_code,
        train_geo_latent_code=train_geo_latent_code,
        cano_view_dirs=cano_view_dirs,
        near_surface_samples=near_surface_samples,
        far_surface_samples=far_surface_samples,
        n_steps=n_steps,
        train_skinning_net=train_skinning_net,
        render_last_pt=render_last_pt,
        pose_input_noise=pose_input_noise,
        view_input_noise=view_input_noise,
        nv_noise_type=nv_noise_type,
        **kwargs
    )

    return model
