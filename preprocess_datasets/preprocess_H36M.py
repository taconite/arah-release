import os
import torch
# import trimesh
import glob
import json
import shutil
import argparse

import numpy as np
import preprocess_datasets.easymocap.mytools.camera_utils as cam_utils

from scipy.spatial.transform import Rotation

from human_body_prior.body_model.body_model import BodyModel

from preprocess_datasets.easymocap.smplmodel import load_model

parser = argparse.ArgumentParser(
    description='Preprocessing for H36M.'
)
parser.add_argument('--data-dir', type=str, help='Directory that contains H36M data.')
parser.add_argument('--out-dir', type=str, help='Directory where preprocessed data is saved.')
parser.add_argument('--seqname', type=str, default='S9', help='Sequence to process.')

n_frames = {'S1': 199, 'S5': 327, 'S6': 233, 'S7': 500, 'S8': 337, 'S9': 393, 'S11': 282}   # statistics from Ani-NeRF paper

if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name, 'Posing')
    out_dir = os.path.join(args.out_dir, seq_name, 'Posing')

    annots = np.load(os.path.join(data_dir, 'annots.npy'), allow_pickle=True).item()
    cameras = annots['cams']

    smpl_dir = os.path.join(data_dir, 'new_params')
    verts_dir = os.path.join(data_dir, 'new_vertices')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1).cuda()

    faces = np.load('body_models/misc/faces.npz')['faces']

    cam_names = []
    for im_path in annots['ims'][0]['ims']:
        cam_names.append(im_path.split('/')[0])

    print (cam_names)

    all_cam_params = {'all_cam_names': cam_names}
    smpl_out_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(smpl_out_dir):
        os.makedirs(smpl_out_dir)

    for cam_idx, cam_name in enumerate(cam_names):
        K = cameras['K'][cam_idx].tolist()
        D = cameras['D'][cam_idx].tolist()
        R = cameras['R'][cam_idx].tolist()

        R_np = np.array(R)
        T = cameras['T'][cam_idx]
        T_np = np.array(T).reshape(3, 1) / 1000.0
        T = T_np.tolist()

        cam_out_dir = os.path.join(out_dir, cam_name)
        if not os.path.exists(cam_out_dir):
            os.makedirs(cam_out_dir)

        img_in_dir = os.path.join(data_dir, cam_name)
        mask_in_dir = os.path.join(data_dir, 'mask_cihp', cam_name)

        img_files = sorted(glob.glob(os.path.join(img_in_dir, '*.jpg')))[:n_frames[seq_name]*5:5]

        cam_params = {'K': K, 'D': D, 'R': R, 'T': T}
        all_cam_params.update({cam_name: cam_params})

        for img_file in img_files:
            print ('Processing: {}'.format(img_file))
            idx = int(os.path.basename(img_file)[:-4])
            frame_index = idx

            mask_file = os.path.join(mask_in_dir, os.path.basename(img_file)[:-4] + '.png')
            smpl_file = os.path.join(smpl_dir, '{}.npy'.format(idx))
            verts_file = os.path.join(verts_dir, '{}.npy'.format(idx))

            if not os.path.exists(smpl_file):
                print ('Cannot find SMPL file for {}: {}, skipping'.format(img_file, smpl_file))
                continue

            # We only process SMPL parameters in world coordinate
            if cam_idx == 0:
                params = np.load(smpl_file, allow_pickle=True).item()

                root_orient = Rotation.from_rotvec(np.array(params['Rh']).reshape([-1])).as_matrix()
                trans = np.array(params['Th']).reshape([3, 1])

                betas = np.array(params['shapes'], dtype=np.float32)
                poses = np.array(params['poses'], dtype=np.float32)
                pose_body = poses[:, 3:66].copy()
                pose_hand = poses[:, 66:].copy()

                poses_torch = torch.from_numpy(poses).cuda()
                pose_body_torch = torch.from_numpy(pose_body).cuda()
                pose_hand_torch = torch.from_numpy(pose_hand).cuda()
                betas_torch = torch.from_numpy(betas).cuda()

                # new_root_orient = Rotation.from_matrix(np.dot(R, root_orient)).as_rotvec().reshape([1, 3]).astype(np.float32)
                # new_trans = (np.dot(R_np, trans) + T_np).reshape([1, 3]).astype(np.float32)
                new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
                new_trans = trans.reshape([1, 3]).astype(np.float32)

                new_root_orient_torch = torch.from_numpy(new_root_orient).cuda()
                new_trans_torch = torch.from_numpy(new_trans).cuda()

                # Get shape vertices
                body = body_model(betas=betas_torch)
                minimal_shape = body.v.detach().cpu().numpy()[0]

                # Get bone transforms
                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)

                body_model_em = load_model(gender='neutral', model_type='smpl')
                verts = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()

                vertices = body.v.detach().cpu().numpy()[0]
                new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)
                new_trans_torch = torch.from_numpy(new_trans).cuda()

                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)

                # # Visualize SMPL mesh
                # vertices = body.v.detach().cpu().numpy()[0]
                # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                # out_filename = os.path.join(smpl_out_dir, '{:06d}.ply'.format(idx))
                # mesh.export(out_filename)

                bone_transforms = body.bone_transforms.detach().cpu().numpy()
                Jtr_posed = body.Jtr.detach().cpu().numpy()

                out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(idx))

                np.savez(out_filename,
                         minimal_shape=minimal_shape,
                         betas=betas,
                         Jtr_posed=Jtr_posed[0],
                         bone_transforms=bone_transforms[0],
                         trans=new_trans[0],
                         root_orient=new_root_orient[0],
                         pose_body=pose_body[0],
                         pose_hand=pose_hand[0])

            shutil.copy(os.path.join(img_file), os.path.join(cam_out_dir, '{:06d}.jpg'.format(idx)))
            shutil.copy(os.path.join(mask_file), os.path.join(cam_out_dir, '{:06d}.png'.format(idx)))

    with open(os.path.join(out_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)
