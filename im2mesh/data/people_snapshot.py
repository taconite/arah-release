import os
import glob
import cv2
import numpy as np
import pickle as pkl
# import logging
import trimesh
import numbers

import igl

from torch.utils import data
from scipy.spatial.transform import Rotation

from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh.utils.utils import get_bound_2d_mask, get_near_far, get_02v_bone_transforms


class PeopleSnapshotDataset(data.Dataset):
    ''' ZJU MoCap dataset class.
    '''

    def __init__(self, dataset_folder,
                 subjects=['female-3-casual'],
                 mode='train',
                 img_size=(1080, 1080),
                 num_fg_samples=1024,
                 num_bg_samples=1024,
                 sampling_rate=1,
                 start_frame=0,
                 end_frame=-1,
                 off_surface_thr=0.2,
                 inside_thr=0.001,
                 box_margin=0.05,
                 sampling='default',
                 sample_reg_surface=False,
                 sample_inside=False,
                 erode_mask=True):
        ''' Initialization of the the People-Snapshot dataset.

        Args:
            dataset_folder (str): dataset folder

            subjects (list of strs): which subjects to use
            mode (str): mode of the dataset. Can be either 'train', 'val' or 'test'
            img_size (int or tuple of ints): target image size we want to sample frome
            num_fg_samples (int): number of points to sample from foreground
            num_bg_samples (int): number of points to sample from background
            sampling_rate (int): sampling rate for video frames
            start_frame (int): start frame of the video
            end_frame (int): end frame of the video
            off_surface_thr (float): threshold for sampling off-surface point loss (in meters)
            inside_thr (float): threshold for determining which points are inside the canonical SMPL mesh
            box_margin (float): bounding box margin added to SMPL bounding box. This bounding box is used to determine sampling region in an image
            sampling (str): ray-sampling method. For current version of code, only 'default' is throughly tested
            sample_reg_surface (bool): whether to sample points on SMPL surface to compute skinning loss
            sample_inside (bool): whether to sample points inside the canonical SMPL mesh to compute inside point loss
            erode_mask (bool): whether to erode ground-truth foreground masks, such that boundary pixels of masks are ignored
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.num_fg_samples = num_fg_samples
        self.num_bg_samples = num_bg_samples
        self.off_surface_thr = off_surface_thr
        self.inside_thr = inside_thr
        self.sampling = sampling
        self.sample_reg_surface = sample_reg_surface
        self.sample_inside = sample_inside
        self.erode_mask = erode_mask

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

        if isinstance(img_size, numbers.Number):
            self.img_size = (int(img_size), int(img_size))
        else:
            self.img_size = img_size

        self.rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        self.rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()
        self.box_margin = box_margin

        self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

        self.ktree_children = np.array([-1,  4,  5,  6,  7,  8,  9,  10,  11,  -1,  -1,  -1,
            15,  16,  17, -1, 18, 19, 20, 21, 22, 23, -1, -1], dtype=np.int32)

        assert (len(subjects) == 1) # TODO: we only support per-subject training at this point

        with open(os.path.join(dataset_folder, subjects[0], 'camera.pkl'), 'rb') as f:
            camera = pkl.load(f, encoding='latin1')

        K, R, T, D = self.get_KRTD(camera)
        self.K = K
        self.R = R
        self.T = T
        self.D = D

        height = camera['height']
        width = camera['width']

        self.orig_img_size = (height, width)

        self.homo_2d = self.init_grid_homo_2d(img_size[0], img_size[1])

        # Get all data
        self.data = []
        cam_idx = 0
        cam_name = '1'
        for subject in subjects:
            subject_dir = os.path.join(dataset_folder, subject)

            if end_frame > 0:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))[start_frame:end_frame:sampling_rate]
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))[start_frame::sampling_rate]

            img_dir = os.path.join(subject_dir, 'image')
            msk_dir = os.path.join(subject_dir, 'mask')

            img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
            frames = np.arange(len(img_files)).tolist()

            if end_frame > 0:
                img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[start_frame:end_frame:sampling_rate]
                mask_files = sorted(glob.glob(os.path.join(msk_dir, '*.png')))[start_frame:end_frame:sampling_rate]
                frames = frames[start_frame:end_frame:sampling_rate]
            else:
                img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[start_frame::sampling_rate]
                mask_files = sorted(glob.glob(os.path.join(msk_dir, '*.png')))[start_frame::sampling_rate]
                frames = frames[start_frame::sampling_rate]

            assert (len(model_files) == len(img_files) and len(mask_files) == len(img_files))

            for d_idx, (f_idx, img_file, mask_file, model_file) in enumerate(zip(frames, img_files, mask_files, model_files)):
                self.data.append(
                        {'subject': subject,
                         'gender': 'female' if 'female' in subject else 'male',
                         'cam_idx': cam_idx,
                         'cam_name': cam_name,
                         'frame_idx': f_idx,
                         'data_idx': d_idx,
                         'img_file': img_file,
                         'mask_file': mask_file,
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

    def get_KRTD(self, camera):
        K = np.zeros([3, 3], dtype=np.float32)
        K[0, 0] = camera['camera_f'][0]
        K[1, 1] = camera['camera_f'][1]
        K[:2, 2] = camera['camera_c']
        K[2, 2] = 1
        R = np.eye(3, dtype=np.float32)
        T = np.zeros([3], dtype=np.float32)
        D = camera['camera_k']

        return K, R, T, D

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
        img_path = self.data[idx]['img_file']
        mask_path = self.data[idx]['mask_file']
        cam_idx = self.data[idx]['cam_idx']
        frame_idx = self.data[idx]['frame_idx']
        data_idx = self.data[idx]['data_idx']
        gender = self.data[idx]['gender']
        data = {}

        # Load and undistort image and mask
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_erode = self.get_mask(mask)

        K = self.K.copy()
        dist = self.D.copy()
        R = self.R.copy()
        cam_trans = self.T.copy()

        cam_loc = self.get_camera_location(R, cam_trans)

        image = cv2.undistort(image, K, dist, None)
        mask = cv2.undistort(mask, K, dist, None)
        mask_erode = cv2.undistort(mask_erode, K, dist, None)

        orig_img_size = (image.shape[0], image.shape[1])

        # Resize image
        img_crop = cv2.resize(image, (self.img_size[1],  self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask, (self.img_size[1],  self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        mask_erode_crop = cv2.resize(mask_erode, (self.img_size[1],  self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        img_crop = img_crop.astype(np.float32)

        img_crop /= 255.0

        side = max(orig_img_size)

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

        # Minimally clothed shape with pose-blend-shapes
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

        fg_sample_mask = mask_erode_crop == 1
        bg_sample_mask = mask_erode_crop == 0
        if self.mode == 'train':
            # Get foreground mask bounding box from which to sample rays
            min_xyz = np.min(minimal_body_vertices, axis=0)
            max_xyz = np.max(minimal_body_vertices, axis=0)
            min_xyz -= self.box_margin
            max_xyz += self.box_margin

            bounds = np.stack([min_xyz, max_xyz], axis=0)
            bound_mask = get_bound_2d_mask(bounds, K, np.concatenate([R, cam_trans.reshape([3, 1])], axis=-1), self.img_size[0], self.img_size[1])
            y_inds_bbox, x_inds_bbox = np.where(bound_mask != 0)

            if self.sampling == 'default':
                # Default sampling strategy: sample specified number of foreground/background pixels
                # Note that for foreground/background we sample an additional 1024 pixels in case some
                # pixels are near the SMPL bounding box boundary; for those boundary points, finding
                # intersections between rays and the SMPL bounding box can fail (i.e. the resulting intersections
                # could have near > far). Eventually we only sample valid num_fg_samples/num_bg_samples pixels

                # Sample foreground pixels
                y_inds, x_inds = np.where(fg_sample_mask)
                fg_inds = np.random.choice(x_inds.shape[0], size=self.num_fg_samples + 1024, replace=False)
                y_inds, x_inds = y_inds[fg_inds], x_inds[fg_inds]
                fg_pixels = img_crop[y_inds, x_inds, :].copy()
                fg_mask = mask_crop[y_inds, x_inds].copy()
                fg_mask_erode = mask_erode_crop[y_inds, x_inds].copy()
                fg_uv = np.dot(self.homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)

                # Sample background pixels
                inds_mask = bg_sample_mask[y_inds_bbox, x_inds_bbox]
                y_inds = y_inds_bbox[inds_mask]
                x_inds = x_inds_bbox[inds_mask]
                bg_inds = np.random.choice(x_inds.shape[0], size=self.num_bg_samples + 1024, replace=False)
                y_inds, x_inds = y_inds[bg_inds], x_inds[bg_inds]
                bg_pixels = np.zeros([x_inds.shape[0], 3], dtype=np.float32)
                bg_mask = mask_crop[y_inds, x_inds].copy()
                bg_mask_erode = mask_erode_crop[y_inds, x_inds].copy()
                bg_uv = np.dot(self.homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)

                sampled_pixels = np.concatenate([fg_pixels, bg_pixels], axis=0)
                sampled_mask = np.concatenate([fg_mask, bg_mask], axis=0) != 0
                sampled_mask_erode = np.concatenate([fg_mask_erode, bg_mask_erode], axis=0) != 0
                sampled_uv = np.concatenate([fg_uv, bg_uv], axis=0)
                sampled_rays_cam = self.normalize_vectors(sampled_uv)
                sampled_rays = self.get_camera_rays(R, sampled_uv)

                near, far, mask_at_box = get_near_far(bounds, np.broadcast_to(cam_loc, sampled_rays.shape), sampled_rays)

                # Now sample num_fg_samples/num_bg_samples pixels where mask_at_box equals to 1
                valid_inds = np.where(mask_at_box[:self.num_fg_samples + 1024] == 1)[0]
                fg_inds = np.random.choice(valid_inds.shape[0], size=self.num_fg_samples, replace=False)
                fg_inds = valid_inds[fg_inds]

                valid_inds = np.where(mask_at_box[self.num_fg_samples + 1024:] == 1)[0] + self.num_fg_samples + 1024
                bg_inds = np.random.choice(valid_inds.shape[0], size=self.num_bg_samples, replace=False)
                bg_inds = valid_inds[bg_inds]

                valid_inds = np.concatenate([fg_inds, bg_inds], axis=-1)

                sampled_pixels = sampled_pixels[valid_inds, ...]
                sampled_mask = sampled_mask[valid_inds, ...]
                sampled_mask_erode = sampled_mask_erode[valid_inds, ...]
                sampled_uv = sampled_uv[valid_inds, ...]
                sampled_rays_cam = sampled_rays_cam[valid_inds, ...]
                sampled_rays = sampled_rays[valid_inds, ...]
                sampled_near = near[valid_inds]
                sampled_far = far[valid_inds]
                sampled_bounds_intersections = np.stack([sampled_near, sampled_far], axis=-1)
            else:
                raise ValueError('Sampling strategy {} is not supported!'.format(self.sampling))
        else:
            # Test/validation mode
            # Get foreground mask bounding box from which to sample rays
            min_xyz = np.min(minimal_body_vertices, axis=0)
            max_xyz = np.max(minimal_body_vertices, axis=0)
            min_xyz -= self.box_margin
            max_xyz += self.box_margin

            bounds = np.stack([min_xyz, max_xyz], axis=0)
            bound_mask = get_bound_2d_mask(bounds, K, np.concatenate([R, cam_trans.reshape([3, 1])], axis=-1), self.img_size[0], self.img_size[1])
            y_inds, x_inds = np.where(bound_mask != 0)

            sampled_pixels = img_crop[y_inds, x_inds, :].copy()
            sampled_mask = np.ones(sampled_pixels.shape[0], dtype=bool)
            sampled_mask_erode = np.ones(sampled_pixels.shape[0], dtype=bool)
            sampled_bg_mask = bg_sample_mask[y_inds, x_inds].copy()
            sampled_pixels[sampled_bg_mask] = 0
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

            image_mask = np.zeros(mask_crop.shape, dtype=bool)
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

        # Sample regularization points
        smpl_mesh = trimesh.Trimesh(vertices=minimal_shape_v, faces=self.faces)
        if self.sample_reg_surface:
            points_uniform = np.random.rand(4096, 3) * 2.0 - 1.0
            query_points = self.unnormalize_canonical_points(points_uniform, coord_min, coord_max, center)
            occupancies = check_mesh_contains(smpl_mesh, query_points)

            points_skinning, _ = smpl_mesh.sample(1024, return_index=True)
            all_points = np.concatenate([query_points, points_skinning], axis=0)
            closest_dists, closest_faces, closest_points = igl.point_mesh_squared_distance(all_points, minimal_shape_v, self.faces)
            points_uniform = points_uniform[(~occupancies) & (closest_dists[:4096, ...] > self.off_surface_thr)]
            if points_uniform.shape[0] >= 1024:
                rand_inds = np.random.choice(points_uniform.shape[0], size=1024, replace=False)
            else:
                rand_inds = np.random.choice(points_uniform.shape[0], size=1024, replace=True)

            points_uniform = points_uniform[rand_inds, :]

            closest_dists = closest_dists[4096:, ...]
            closest_faces = closest_faces[4096:, ...]
            closest_points = closest_points[4096:, ...]
            bary_coords = igl.barycentric_coordinates_tri(
                    closest_points,
                    minimal_shape_v[self.faces[closest_faces, 0], :],
                    minimal_shape_v[self.faces[closest_faces, 1], :],
                    minimal_shape_v[self.faces[closest_faces, 2], :]
            )
            vert_ids = self.faces[closest_faces, ...]
            pts_W = (skinning_weights[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)
            # points_skinning = self.normalize_canonical_points(points_skinning, coord_min, coord_max, center)
        else:
            points_uniform = np.random.rand(4096, 3) * 2.0 - 1.0
            query_points = self.unnormalize_canonical_points(points_uniform, coord_min, coord_max, center)
            occupancies = check_mesh_contains(smpl_mesh, query_points)
            closest_dists, _, _ = igl.point_mesh_squared_distance(query_points, minimal_shape_v, self.faces)
            points_uniform = points_uniform[(~occupancies) & (closest_dists > self.off_surface_thr)]
            if points_uniform.shape[0] >= 1024:
                rand_inds = np.random.choice(points_uniform.shape[0], size=1024, replace=False)
            else:
                rand_inds = np.random.choice(points_uniform.shape[0], size=1024, replace=True)

            points_uniform = points_uniform[rand_inds, :]

            points_skinning = np.zeros([24, 3], dtype=np.float32)
            pts_W = np.zeros([24, 24], dtype=np.float32)
            for j_idx in range(24):
                points_skinning[j_idx, :] = np.mean(minimal_shape_v[part_idx == j_idx, :], axis=0)
                pts_W[j_idx, j_idx] = 1.0


        if self.sample_inside:
            inside_Jtr_points = np.zeros([22, 3], dtype=np.float32)
            for j_idx in range(22):
                inside_Jtr_points[j_idx, :] = np.mean(minimal_shape_v[part_idx == j_idx, :], axis=0)

            # Sample points that are inside SMPL mesh, excluding hands
            inside_points, face_idx = smpl_mesh.sample(4096, return_index=True)
            inside_points += np.random.normal(scale=0.5, size=inside_points.shape)
            occupancies = check_mesh_contains(smpl_mesh, inside_points)
            inside_points = inside_points[occupancies]
            closest_dists, closest_faces, closest_points = igl.point_mesh_squared_distance(inside_points, minimal_shape_v, self.faces)
            bary_coords = igl.barycentric_coordinates_tri(
                    closest_points,
                    minimal_shape_v[self.faces[closest_faces, 0], :],
                    minimal_shape_v[self.faces[closest_faces, 1], :],
                    minimal_shape_v[self.faces[closest_faces, 2], :]
            )
            vert_ids = self.faces[closest_faces, ...]
            pts_W_ = (skinning_weights[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)
            part_idx = pts_W_.argmax(-1)
            inside_points = inside_points[(part_idx != 22) & (part_idx != 23) & (closest_dists >= self.inside_thr), :]

            if len(inside_points) > 0:
                inside_points = np.concatenate([inside_points, inside_Jtr_points], axis=0)
            else:
                inside_points = inside_Jtr_points

            if inside_points.shape[0] >= 1024:
                rand_inds = np.random.choice(inside_points.shape[0], size=1024, replace=False)
            else:
                rand_inds = np.random.choice(inside_points.shape[0], size=1024, replace=True)

            inside_points = inside_points[rand_inds, :]
            inside_points = self.normalize_canonical_points(inside_points, coord_min, coord_max, center)

        data = {
            'trans': trans,
            'bone_transforms': bone_transforms.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v.astype(np.float32),
            'coord_max': coord_max.astype(np.float32),
            'coord_min': coord_min.astype(np.float32),
            'center': center.astype(np.float32),
            'minimal_shape': minimal_shape_v.astype(np.float32),
            'smpl_vertices': minimal_body_vertices.astype(np.float32),
            'points_skinning': points_skinning.astype(np.float32),
            'skinning_weights': skinning_weights.astype(np.float32),
            'sampled_weights': pts_W.astype(np.float32),
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
            'points_uniform': points_uniform.astype(np.float32),
        }

        if self.sample_inside:
            data.update({'points_inside': inside_points.astype(np.float32)})

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
