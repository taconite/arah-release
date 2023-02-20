import os
import glob
import cv2
import numpy as np
import json
# import logging
import numbers

from torch.utils import data
from scipy.spatial.transform import Rotation

from im2mesh.utils.utils import get_bound_2d_mask, get_near_far, get_02v_bone_transforms


class ZJUMOCAPODPDataset(data.Dataset):
    ''' ZJU MoCap dataset class for out-of-distribution poses.
    '''

    def __init__(self, dataset_folder,
                 subjects=['CoreView_313'],
                 pose_dir='MPI-Limits_03009_op8_poses',
                 mode='test',
                 orig_img_size=(1024, 1024),
                 img_size=(512, 512),
                 num_fg_samples=1024,
                 num_bg_samples=1024,
                 sampling_rate=1,
                 start_frame=0,
                 end_frame=-1,
                 views=[],
                 box_margin=0.05):
        ''' Initialization of the ZJU-MoCap with out-of-distribution poses dataset.

        Args:
            dataset_folder (str): dataset folder
            subjects (list of strs): which subjects to use
            pose_dir (str): name of the motion sequence to use
            mode (str): mode of the dataset. Can be either 'train', 'val' or 'test'
            orig_img_size (int or tuple of ints): original image on which the model was trained
            img_size (int or tuple of ints): target image size we want to sample frome
            num_fg_samples (int): number of points to sample from foreground
            num_bg_samples (int): number of points to sample from background
            sampling_rate (int): sampling rate for video frames
            start_frame (int): start frame of the video
            end_frame (int): end frame of the video
            views (list of strs): which views to use
            box_margin (float): bounding box margin added to SMPL bounding box. This bounding box is used to determine sampling region in an image
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.num_fg_samples = num_fg_samples
        self.num_bg_samples = num_bg_samples

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

        if isinstance(img_size, numbers.Number):
            self.img_size = (int(img_size), int(img_size))
        else:
            self.img_size = img_size

        if isinstance(orig_img_size, numbers.Number):
            self.orig_img_size = (int(orig_img_size), int(orig_img_size))
        else:
            self.orig_img_size = orig_img_size

        self.rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        self.rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()
        self.box_margin = box_margin

        self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

        self.ktree_children = np.array([-1,  4,  5,  6,  7,  8,  9,  10,  11,  -1,  -1,  -1,
            15,  16,  17, -1, 18, 19, 20, 21, 22, 23, -1, -1], dtype=np.int32)

        assert (len(subjects) == 1) # TODO: we only support per-subject training at this point

        with open(os.path.join(dataset_folder, subjects[0], 'cam_params.json'), 'r') as f:
            cameras = json.load(f)

        self.cameras = cameras

        if len(views) == 0:
            cam_names = cameras['all_cam_names']
        else:
            cam_names = views

        self.cam_names = cam_names

        self.homo_2d = self.init_grid_homo_2d(img_size[0], img_size[1])

        # Get all data
        self.data = []
        for subject in subjects:
            subject_dir = os.path.join(dataset_folder, subject)

            model_files = sorted(glob.glob(os.path.join(subject_dir, pose_dir, '*.npz')))
            frames = np.arange(len(model_files)).tolist()
            if end_frame > 0:
                model_files = sorted(glob.glob(os.path.join(subject_dir, pose_dir, '*.npz')))[start_frame:end_frame:sampling_rate]
                frames = frames[start_frame:end_frame:sampling_rate]
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, pose_dir, '*.npz')))[start_frame::sampling_rate]
                frames = frames[start_frame::sampling_rate]

            for cam_idx, cam_name in enumerate(cam_names):
                for d_idx, (f_idx, model_file) in enumerate(zip(frames, model_files)):
                    self.data.append(
                            {'subject': subject,
                             'gender': 'neutral',
                             'cam_idx': cam_idx,
                             'cam_name': cam_name,
                             'frame_idx': f_idx,
                             'data_idx': d_idx,
                             'model_file': model_file}
                        )

    def unnormalize_canonical_points(self, pts, coord_min, coord_max, center):
        padding = (coord_max - coord_min) * 0.05
        pts = (pts / 2.0 + 0.5) * 1.1 * (coord_max - coord_min) + coord_min - padding +  center

        return pts

    def normalize_canonical_points(self, pts, coord_min, coord_max, center):
        pts -= center
        padding = (coord_max - coord_min) * 0.05
        pts = (pts - coord_min + padding) / (coord_max - coord_min) / 1.1
        pts -= 0.5
        pts *= 2.

        return pts

    def get_meshgrid(self, height, width):
        Y, X = np.meshgrid(np.arange(height, dtype=np.float32),
                           np.arange(width, dtype=np.float32),
                           indexing='ij'
                          )
        grid_map = np.stack([X, Y], axis=-1)  # (height, width, 2)

        return grid_map

    def get_homo_2d_from_xy(self, xy):
        H, W = xy.shape[0], xy.shape[1]
        homo_ones = np.ones((H, W, 1), dtype=np.float32)

        homo_2d = np.concatenate((xy, homo_ones), axis=2)
        return homo_2d

    def get_homo_2d(self, height, width):
        xy = self.get_meshgrid(height, width)
        homo_2d = self.get_homo_2d_from_xy(xy)

        return homo_2d

    def init_grid_homo_2d(self, height, width):
        homo_2d = self.get_homo_2d(height, width)
        homo_2d = homo_2d    # (height*width, 3)

        return homo_2d

    def normalize_vectors(self, x):
        norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        eps = 1e-12
        x = x / (norm + eps)
        return x

    def get_camera_location(self, R, t):
        cam_loc = np.dot(-R.T, t)
        return cam_loc

    def get_camera_rays(self, R, homo_2d):
        rays = np.dot(homo_2d, R) # (H*W, 3)
        rays = self.normalize_vectors(rays) # (H*W, 3)
        return rays

    def get_mask(self, mask_in):
        mask = (mask_in != 0).astype(np.uint8)

        if self.erode_mask or self.mode in ['val', 'test']:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            mask_erode = cv2.erode(mask.copy(), kernel)
            mask_dilate = cv2.dilate(mask.copy(), kernel)
            mask[(mask_dilate - mask_erode) == 1] = 100

        return mask

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data_path = self.data[idx]['model_file']
        cam_name = self.data[idx]['cam_name']
        cam_idx = self.data[idx]['cam_idx']
        frame_idx = self.data[idx]['frame_idx']
        data_idx = self.data[idx]['data_idx']
        gender = self.data[idx]['gender']
        data = {}

        K = np.array(self.cameras[cam_name]['K'], dtype=np.float32)
        R = np.array(self.cameras[cam_name]['R'], np.float32)
        cam_trans = np.array(self.cameras[cam_name]['T'], np.float32).ravel()

        cam_loc = self.get_camera_location(R, cam_trans)

        # Dummy placeholders for image and mask
        img_crop = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.float32)
        mask_crop = np.zeros((self.img_size[1], self.img_size[0]), dtype=bool)

        side = max(self.orig_img_size)

        # Update camera parameters
        principal_point = K[:2, -1].reshape(-1).astype(np.float32)
        focal_length = np.array([K[0, 0], K[1, 1]], dtype=np.float32)

        focal_length = focal_length / side  * max(self.img_size)
        principal_point = principal_point / side * max(self.img_size)

        K[:2, -1] = principal_point
        K[0, 0] = focal_length[0]
        K[1, 1] = focal_length[1]

        K_inv = np.linalg.inv(K)    # for mapping rays from camera space to world space

        # 3D models and points
        model_dict = np.load(data_path)

        trans = model_dict['trans'].astype(np.float32)
        minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        n_smpl_points = minimal_shape.shape[0]
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose_body = model_dict['pose_body'].astype(np.float32)
        pose_hand = model_dict['pose_hand'].astype(np.float32)
        Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)
        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        pose_mat_full = pose.as_matrix()       # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()    # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape([-1, 9])   # 24 x 9, root rotation is set to identity

        pose_rot_full = pose_mat_full.reshape([-1, 9])   # 24 x 9, including root rotation

        # Minimally clothed shape
        posedir = self.posedirs[gender]
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        ident = np.eye(3)
        pose_feature = (pose_mat - ident).reshape([207, 1])
        pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
        minimal_shape += pose_offsets

        # Get posed minimally-clothed shape
        skinning_weights = self.skinning_weights[gender]
        T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([minimal_shape, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        minimal_body_vertices = (np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans).astype(np.float32)

        if self.mode in ['train', 'val']:
            raise ValueError('Only test mode is supported!')
        else:
            # Get foreground mask bounding box from which to sample rays
            min_xyz = np.min(minimal_body_vertices, axis=0)
            max_xyz = np.max(minimal_body_vertices, axis=0)
            min_xyz -= self.box_margin
            max_xyz += self.box_margin

            bounds = np.stack([min_xyz, max_xyz], axis=0)
            bound_mask = get_bound_2d_mask(bounds, K, np.concatenate([R, cam_trans.reshape([3, 1])], axis=-1), self.img_size[0], self.img_size[1])
            y_inds, x_inds = np.where(bound_mask != 0)

            sampled_pixels = img_crop[y_inds, x_inds, :].copy() # just placeholder
            sampled_mask = np.ones(sampled_pixels.shape[0], dtype=bool)
            sampled_mask_erode = np.ones(sampled_pixels.shape[0], dtype=bool)
            sampled_uv = np.dot(self.homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)
            sampled_rays_cam = self.normalize_vectors(sampled_uv)
            sampled_rays = self.get_camera_rays(R, sampled_uv)

            near, far, mask_at_box = get_near_far(bounds, np.broadcast_to(cam_loc, sampled_rays.shape), sampled_rays)

            sampled_pixels = sampled_pixels[mask_at_box, ...]
            sampled_mask = sampled_mask[mask_at_box, ...]
            sampled_mask_erode = sampled_mask_erode[mask_at_box, ...]
            sampled_uv = sampled_uv[mask_at_box, ...]
            sampled_rays_cam = sampled_rays_cam[mask_at_box, ...]
            sampled_rays = sampled_rays[mask_at_box, ...]
            sampled_near = near[mask_at_box]
            sampled_far = far[mask_at_box]
            sampled_bounds_intersections = np.stack([sampled_near, sampled_far], axis=-1)

            image_mask = np.zeros_like(mask_crop)
            image_mask[y_inds[mask_at_box], x_inds[mask_at_box]] = True

        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = get_02v_bone_transforms(Jtr, self.rot45p, self.rot45n)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        minimal_shape_v = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        # Normalize conanical pose points with GT full-body scales. This should be fine as
        # at test time we register each frame first, thus obtaining full-body scale
        center = np.mean(minimal_shape_v, axis=0)
        minimal_shape_v_centered = minimal_shape_v - center
        coord_max = minimal_shape_v_centered.max()
        coord_min = minimal_shape_v_centered.min()

        padding = (coord_max - coord_min) * 0.05

        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - coord_min + padding) / (coord_max - coord_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # Get centroid of each part
        Jtr_mid = np.zeros([24, 3], dtype=np.float32)
        part_idx = skinning_weights.argmax(-1)
        for j_idx in range(24):
            Jtr_mid[j_idx, :] = np.mean(minimal_body_vertices[part_idx == j_idx, :], axis=0)

        data = {
            'trans': trans,
            'bone_transforms': bone_transforms.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v.astype(np.float32),
            'coord_max': coord_max.astype(np.float32),
            'coord_min': coord_min.astype(np.float32),
            'center': center.astype(np.float32),
            'minimal_shape': minimal_shape_v.astype(np.float32),
            'smpl_vertices': minimal_body_vertices.astype(np.float32),
            'skinning_weights': skinning_weights.astype(np.float32),
            'root_orient': root_orient,
            'pose_hand': pose_hand,
            'pose_body': pose_body,
            'Jtr_mid': Jtr_mid,
            'rots': pose_rot.astype(np.float32),
            'Jtrs': Jtr_norm.astype(np.float32),
            'rots_full': pose_rot_full.astype(np.float32),
            'Jtrs_posed': Jtr_posed.astype(np.float32),
            'center_cam': principal_point,
            'focal_length': focal_length,
            'K': K,
            'R': R,
            'T': cam_trans,
            'cam_loc': cam_loc,
        }

        data_out = {}
        field_name = 'image'
        for k, v in data.items():
            if k is None:
                data_out[field_name] = v
            else:
                data_out['%s.%s' % (field_name, k)] = v

        data_out.update(
            {'inputs': sampled_pixels,
             'inputs.mask': sampled_mask,
             'inputs.mask_erode': sampled_mask_erode,
             'inputs.uv': sampled_uv,
             'inputs.ray_dirs': sampled_rays,
             'inputs.ray_dirs_cam': sampled_rays_cam,
             'inputs.body_bounds_intersections': sampled_bounds_intersections,
             'inputs.gender': gender,
             'inputs.img_height': int(self.img_size[0]),
             'inputs.img_width': int(self.img_size[1]),
             'inputs.cam_idx': int(cam_idx),
             'inputs.frame_idx': int(frame_idx),
             'inputs.data_idx': int(data_idx),
             'idx': int(idx),
             'inputs.novel_seq': True,
            }
        )

        if self.mode != 'train':
            data_out.update(
                {'inputs.image_mask': image_mask,
                }
            )

        return data_out

    def get_model_dict(self, idx):
        return self.data[idx]
