import os
import shutil
import json
import argparse
import torch

import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation
from human_body_prior.body_model.body_model import BodyModel

parser = argparse.ArgumentParser('Preprocessing of AIST++.')
# Information for AIST++. This is for extracting unseen pose vectors
parser.add_argument('--data-dir', type=str, default='/home/sfwang/Datasets/AIST++', help='Directory that contains all AIST++ .pkl data.')
parser.add_argument('--out-dir', type=str, default='data/odp', help='Directory where preprocessed data is saved.')
parser.add_argument('--seqname', type=str, default='gBR_sBM_cAll_d04_mBR1_ch05', help='Sequence to process.')
# Information for input dataset. This is for extracting body shape and camera parameters of the trained model
parser.add_argument('--in-dataset', type=str, default='data/zju_mocap', help='Input dataset to process.')
parser.add_argument('--subject', type=str, default='CoreView_313', help='Which subject to use.')
parser.add_argument('--view', type=int, default=1, help='Which view to use.')

if __name__ == '__main__':
    args = parser.parse_args()
    view = str(args.view)

    aist_name = args.seqname
    input_file = '{}/{}.pkl'.format(args.data_dir, aist_name)
    with open(input_file, 'rb') as f:
        input_data = pkl.load(f)

    additional_R = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix().astype(np.float32) # neccesary to align SMPL with image

    body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1).cuda()

    with open('{}/{}/cam_params.json'.format(args.in_dataset, args.subject), 'r') as f:
        cameras = json.load(f)

    K = np.array(cameras[view]['K'], dtype=np.float32)
    dist = np.array(cameras[view]['D'], dtype=np.float32).ravel()
    R = np.array(cameras[view]['R'], np.float32)
    cam_trans = np.array(cameras[view]['T'], np.float32).ravel()

    # Extract information for training subject
    data = np.load('{}/{}/models/000001.npz'.format(args.in_dataset, args.subject))
    tgt_root_orient = Rotation.from_rotvec(data['root_orient'])
    tgt_root_mat = tgt_root_orient.as_matrix()
    tgt_trans = data['trans'].copy()
    tgt_Jtr = data['Jtr_posed'].copy()
    tgt_betas = data['betas'].astype(np.float32).copy()
    minimal_shape = data['minimal_shape'].astype(np.float32).copy()

    poses = input_data['smpl_poses'][::2]
    transl = input_data['smpl_trans'][::2] / 100.0  # AIST++ translation is in cm

    root_orient_0 = None
    # trans_0 = None

    seq_name = aist_name
    out_dir = os.path.join(args.out_dir, args.subject, seq_name + '_view{}'.format(view))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    # Copy camera parameters, if haven't done so already
    if not os.path.exists(os.path.join(args.out_dir, args.subject, 'cam_params.json')):
        shutil.copy('{}/{}/cam_params.json'.format(args.in_dataset, args.subject), os.path.join(args.out_dir, args.subject, 'cam_params.json'))

    for cnt, (pose, trans) in enumerate(zip(poses, transl)):
        input_filebase = '{:06d}'.format(cnt)

        out_filename = os.path.join(out_dir, input_filebase)

        pose = pose.astype(np.float32)

        root_orient = pose[:3].copy()
        if cnt == 0:
            root_orient_0 = np.linalg.inv(Rotation.from_rotvec(root_orient).as_matrix())

        # For the rotation, we need to:
        # 1) set current root_orient relative to the initial root_orient
        # 2) apply additional_R to root_orient
        # 3) convert root_orient to world coordinate
        root_orient = Rotation.from_rotvec(root_orient).as_matrix()
        root_orient = R.T @ additional_R @ root_orient_0 @ root_orient

        root_orient = Rotation.from_matrix(root_orient).as_rotvec().astype(np.float32)

        pose_body = pose[3:66].copy()
        pose_hand = pose[66:].copy()

        root_orient_torch = torch.from_numpy(root_orient).unsqueeze(0).cuda()
        pose_body_torch = torch.from_numpy(pose_body).unsqueeze(0).cuda()
        pose_hand_torch = torch.from_numpy(pose_hand).unsqueeze(0).cuda()
        betas_torch = torch.from_numpy(tgt_betas).cuda()

        # For translation, we set it to [0, 0, 2.7] in the camera coordinate,
        # then convert it to the world coordinate
        trans = trans.astype(np.float32)
        # if cnt == 0:
        #     print (trans)
        #     trans_0 = trans.copy()

        trans = np.zeros_like(trans)
        trans[-1] += 2.7    # 2.7 is just a magical number; it may need to be tuned for each sequence to obtain the best relative position for rendering
        trans = np.dot(trans - cam_trans, R)

        trans_torch = torch.from_numpy(trans).unsqueeze(0).cuda()

        # Get and save new parameters
        body = body_model(root_orient=root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=trans_torch)
        bone_transforms = body.bone_transforms.detach().cpu().numpy()[0]
        Jtr_posed = body.Jtr.detach().cpu().numpy()

        np.savez(out_filename,
                 minimal_shape=minimal_shape,
                 betas=tgt_betas,
                 Jtr_posed=Jtr_posed[0],
                 bone_transforms=bone_transforms,
                 trans=trans,
                 root_orient=root_orient,
                 pose_body=pose_body,
                 pose_hand=pose_hand)
