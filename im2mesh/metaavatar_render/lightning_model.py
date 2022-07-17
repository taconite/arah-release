import os
import numbers
import imageio
import shutil
import numpy as np
import lpips
import torch
import torch.nn.functional as F
import kornia.geometry.conversions as conversions
import pytorch_lightning as pl

from im2mesh.metaavatar_render.renderer.loss import IDHRLoss

from im2mesh.utils.root_finding_utils import (
    normalize_canonical_points
)

from im2mesh.utils.eval import (psnr_metric, ssim_metric, lpips_metric)

# Utility functions for computing camera rays
def normalize_vectors(x):
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    eps = 1e-12
    x = x / (norm + eps)
    return x

def get_camera_location(R, t):
    cam_loc = torch.matmul(-R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
    return cam_loc

def get_camera_rays(R, homo_2d):
    rays = torch.matmul(homo_2d, R) # (B, H*W, 3)
    rays = normalize_vectors(rays) # (B, H*W, 3)
    return rays

# Utility functions for Vitruvian A-pose
def get_transforms_02v(Jtr):
    device = Jtr.device

    from scipy.spatial.transform import Rotation as R
    rot45p = torch.tensor(R.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(R.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(24, 1, 1)

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    R_02v_l = []
    t_02v_l = []
    chain = [1, 4, 7, 10]
    rot = rot45p
    for i, j_idx in enumerate(chain):
        R_02v_l.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_l[i-1]

        t_02v_l.append(t)

    R_02v_l = torch.stack(R_02v_l, dim=0)
    t_02v_l = torch.stack(t_02v_l, dim=0)
    t_02v_l = t_02v_l - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_l = F.pad(R_02v_l, (0, 0, 0, 1))  # 4 x 4 x 3
    t_02v_l = F.pad(t_02v_l, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_l, t_02v_l.unsqueeze(-1)], dim=-1)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    R_02v_r = []
    t_02v_r = []
    chain = [2, 5, 8, 11]
    rot = rot45n
    for i, j_idx in enumerate(chain):
        # bone_transforms_02v[j_idx, :3, :3] = rot
        R_02v_r.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_r[i-1]

        t_02v_r.append(t)

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    R_02v_r = torch.stack(R_02v_r, dim=0)
    t_02v_r = torch.stack(t_02v_r, dim=0)
    t_02v_r = t_02v_r - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_r = F.pad(R_02v_r, (0, 0, 0, 1))  # 4 x 3
    t_02v_r = F.pad(t_02v_r, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_r, t_02v_r.unsqueeze(-1)], dim=-1)

    return bone_transforms_02v

class LightningModel(pl.LightningModule):
    ''' PyTorch Lightning model object for the renderer. '''
    def __init__(self, model, cfg, val_size=None):
        ''' Initialization of the trainer class.

        Args:
            model (nn.Module): MetaAvatar-Render Network model
            cfg (dict): configuration dictionary parsed from .yaml file
        '''
        super().__init__()

        self.model = model
        self.cfg = cfg

        rgb_weight = cfg['training']['rgb_weight']
        perceptual_weight = cfg['training']['perceptual_weight']
        eikonal_weight = cfg['training']['eikonal_weight']
        mask_weight = cfg['training']['mask_weight']
        off_surface_weight = cfg['training']['off_surface_weight']
        inside_weight = cfg['training']['inside_weight']
        params_weight = cfg['training']['params_weight']
        skinning_weight = cfg['training']['skinning_weight']
        rgb_loss_type = cfg['training']['rgb_loss_type']

        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.criteria = IDHRLoss(rgb_weight=rgb_weight,
                                 perceptual_weight=perceptual_weight,
                                 eikonal_weight=eikonal_weight,
                                 mask_weight=mask_weight,
                                 off_surface_weight=off_surface_weight,
                                 inside_weight=inside_weight,
                                 params_weight=params_weight,
                                 skinning_weight=skinning_weight,
                                 rgb_loss_type=rgb_loss_type,
                                 perceptual_loss_fn=self.loss_fn_vgg)

        self.val_size = val_size

    def training_step(self, batch, batch_idx):
        ''' Performs a training step.

        Args:
            batch (dict): data dictionary
            batch_idx (dict): index of the data
        '''
        # self.model.train()

        loss_dict = self.compute_loss(batch)
        loss = loss_dict['loss']

        for k, v in loss_dict.items():
            self.log('train/{}'.format(k), v, sync_dist=True)

        # assert (self.model.deviation_decoder.variance.item() > 0.0)
        self.log('train/deviation', self.model.deviation_decoder.variance.item(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ''' Performs a validation step.

        Args:
            batch (dict): data dictionary
            batch_idx (dict): index of the data
        '''

        inputs = self.compose_inputs(batch, eval=True)

        img_height = batch['inputs.img_height'].item()
        img_width = batch['inputs.img_width'].item()

        with torch.no_grad():
            model_outputs = self.model(inputs, gen_cano_mesh=False, eval=True)

        # Save predicted image
        image_mask = batch.get('inputs.image_mask')
        output_rgb = model_outputs['rgb_values'].reshape(-1, 3)[:image_mask.sum(), :]
        pred_pixels = torch.zeros(1, img_height, img_width, 3, device=self.device, dtype=torch.float32)
        pred_pixels.masked_scatter_(image_mask.view(1, img_height, img_width, 1), output_rgb)
        pred_pixels = pred_pixels.squeeze(0)

        # Render normal image
        if 'output_normal' not in model_outputs.keys():
            points_cam = model_outputs['points_cam'].reshape(-1, 3)[:image_mask.sum(), :]
            pred_points = torch.zeros(1, img_height, img_width, 3, device=self.device, dtype=torch.float32)
            pred_points.masked_scatter_(image_mask.view(1, img_height, img_width, 1), points_cam)
            pred_points = pred_points.squeeze(0)

            zs = pred_points[:, :, 2].clone()
            xs = pred_points[:, :, 0].clone()
            ys = pred_points[:, :, 1].clone()

            zy = (zs[1:, :] - zs[:-1, :]) / (ys[1:, :] - ys[:-1, :])
            zx = (zs[:, 1:] - zs[:, :-1]) / (xs[:, 1:] - xs[:, :-1])
            pred_normals = torch.zeros(img_height, img_width, 3, device=self.device, dtype=torch.float32)
            pred_normals[:-1, :, 1] = -zy
            pred_normals[:, :-1, 0] = -zx
            pred_normals[:, :, 2] = 1

            n = torch.linalg.norm(pred_normals, dim=-1, keepdim=True)
            pred_normals /= n

            pred_normals[pred_normals.isnan()] = -1
            pred_normals = ((pred_normals + 1) / 2.0).clip(0.0, 1.0)
        else:
            pred_normals = model_outputs['output_normal'].squeeze(0)

        # Save GT image
        image = batch.get('inputs')
        gt_rgb = image.reshape(-1, 3)[:image_mask.sum(), :]
        gt_pixels = torch.zeros(1, img_height, img_width, 3, device=self.device, dtype=torch.float32)
        gt_pixels.masked_scatter_(image_mask.view(1, img_height, img_width, 1), gt_rgb)
        gt_pixels = gt_pixels.squeeze(0)

        # Evaluate the generated image with PSNR/SSIM/LPIPS
        eval_dict = {}

        pred_img = model_outputs['rgb_values'].reshape(-1, 3).detach().cpu().numpy()
        gt_img = image.reshape(-1, 3).detach().cpu().numpy()
        bbox_mask = (image_mask).squeeze(0).detach().cpu().numpy()
        eval_dict['psnr'] = psnr_metric(pred_img, gt_img)
        eval_dict['ssim'] = ssim_metric(pred_pixels.detach().cpu().numpy(), gt_pixels.detach().cpu().numpy(), bbox_mask)
        eval_dict['lpips'] = lpips_metric(pred_pixels.detach().cpu().numpy(), gt_pixels.detach().cpu().numpy(), bbox_mask, self.loss_fn_vgg, self.device)

        eval_dict['rgb_pred'] = pred_pixels.permute(2, 0, 1)
        eval_dict['normal_pred'] = pred_normals.permute(2, 0, 1)
        eval_dict['rgb_gt'] = gt_pixels.permute(2, 0, 1)

        return eval_dict

    def _process_validation_epoch_outputs(self, validation_step_outputs):
        psnr, ssim, lpips = [], [], []
        rgb_preds, normal_preds, rgb_gts = [], [], []
        for output in validation_step_outputs:
            psnr.append(output['psnr'])
            ssim.append(output['ssim'])
            lpips.append(output['lpips'])

            rgb_preds.append(output['rgb_pred'])
            normal_preds.append(output['normal_pred'])
            rgb_gts.append(output['rgb_gt'])

        return psnr, ssim, lpips, torch.stack(rgb_preds, dim=0), torch.stack(normal_preds, dim=0), torch.stack(rgb_gts, dim=0)

    def validation_epoch_end(self, validation_step_outputs):

        psnr, ssim, lpips, rgb_pred, normal_pred, rgb_gt = self._process_validation_epoch_outputs(validation_step_outputs)

        psnr = self.all_gather(psnr)
        ssim = self.all_gather(ssim)
        lpips = self.all_gather(lpips)

        rgb_pred = self.all_gather(rgb_pred)
        normal_pred = self.all_gather(normal_pred)
        rgb_gt = self.all_gather(rgb_gt)

        psnr = torch.stack(psnr, dim=0)    # batch_size x world_size
        ssim = torch.stack(ssim, dim=0)    # batch_size x world_size
        lpips = torch.stack(lpips, dim=0)    # batch_size x world_size

        if len(psnr.shape) > 1 and self.global_rank == 0:
            # Only do this for DDP validation
            rgb_pred = rgb_pred.transpose(0, 1)    # batch_size x world_size x ...
            normal_pred = normal_pred.transpose(0, 1)    # batch_size x world_size x ...
            rgb_gt = rgb_gt.transpose(0, 1)    # batch_size x world_size x ...
            if self.val_size is not None:
                # Dirty way to handle the case when the # of GPUs does not evenly divide the size of validation dataset
                psnr = psnr.reshape(-1)[:self.val_size]
                ssim = ssim.reshape(-1)[:self.val_size]
                lpips = lpips.reshape(-1)[:self.val_size]

                rgb_pred = rgb_pred.reshape(-1, *rgb_pred.shape[2:])[:self.val_size, ...]
                normal_pred = normal_pred.reshape(-1, *normal_pred.shape[2:])[:self.val_size, ...]
                rgb_gt = rgb_gt.reshape(-1, *rgb_gt.shape[2:])[:self.val_size, ...]
            else:
                psnr = psnr.reshape(-1)
                ssim = ssim.reshape(-1)
                lpips = lpips.reshape(-1)

                rgb_pred = rgb_pred.reshape(-1, *rgb_pred.shape[2:])
                normal_pred = normal_pred.reshape(-1, *normal_pred.shape[2:])
                rgb_gt = rgb_gt.reshape(-1, *rgb_gt.shape[2:])

            self.log('psnr', psnr.mean(), rank_zero_only=True)
            self.log('ssim', ssim.mean(), rank_zero_only=True)
            self.log('lpips', lpips.mean(), rank_zero_only=True)

            for rgb, normal, gt in zip(rgb_pred, normal_pred, rgb_gt):
                self.logger.log_image(key="validation_samples",
                        images=[rgb, normal, gt],
                        caption=["rgb_pred", "normal_pred", "rgb_gt"]
                )

        elif len(psnr.shape) == 1:
            self.log('psnr', psnr.mean(), rank_zero_only=True)
            self.log('ssim', ssim.mean(), rank_zero_only=True)
            self.log('lpips', lpips.mean(), rank_zero_only=True)

            for rgb, normal, gt in zip(rgb_pred, normal_pred, rgb_gt):
                self.logger.log_image(key="validation_samples",
                        images=[rgb, normal, gt],
                        caption=["rgb_pred", "normal_pred", "rgb_gt"]
                )

    def test_step(self, batch, batch_idx):
        ''' Performs a test step.

        Args:
            batch (dict): data dictionary
            batch_idx (dict): index of the data
        '''

        inputs = self.compose_inputs(batch, eval=True)

        img_height = batch['inputs.img_height'].item()
        img_width = batch['inputs.img_width'].item()

        with torch.no_grad():
            model_outputs = self.model(inputs, gen_cano_mesh=True, eval=True)

        # Save predicted image
        image_mask = batch.get('inputs.image_mask')
        output_rgb = model_outputs['rgb_values'].reshape(-1, 3)[:image_mask.sum(), :]
        pred_pixels = torch.zeros(1, img_height, img_width, 3, device=self.device, dtype=torch.float32)
        pred_pixels.masked_scatter_(image_mask.view(1, img_height, img_width, 1), output_rgb)
        pred_pixels = pred_pixels.squeeze(0)

        # Render normal image
        pred_normals = model_outputs['output_normal'].squeeze(0)
        pred_normals_front = model_outputs['normal_cano_front'].squeeze(0)
        pred_normals_back = model_outputs['normal_cano_back'].squeeze(0)

        # Store result images
        eval_dict = {}
        eval_dict['rgb_pred'] = pred_pixels.permute(2, 0, 1)
        eval_dict['normal_pred'] = pred_normals.permute(2, 0, 1)
        eval_dict['normal_front'] = pred_normals_front.permute(2, 0, 1)
        eval_dict['normal_back'] = pred_normals_back.permute(2, 0, 1)

        return eval_dict

    def _process_test_epoch_outputs(self, validation_step_outputs):
        rgb_preds, normal_preds, normal_front, normal_back = [], [], [], []
        for output in validation_step_outputs:
            rgb_preds.append(output['rgb_pred'])
            normal_preds.append(output['normal_pred'])
            normal_front.append(output['normal_front'])
            normal_back.append(output['normal_back'])

        return torch.stack(rgb_preds, dim=0), torch.stack(normal_preds, dim=0), torch.stack(normal_front, dim=0), torch.stack(normal_back, dim=0)

    def test_epoch_end(self, test_step_outputs):

        rgb_pred, normal_pred, normal_front, normal_back = self._process_test_epoch_outputs(test_step_outputs)

        rgb_pred = self.all_gather(rgb_pred)
        normal_pred = self.all_gather(normal_pred)
        normal_front = self.all_gather(normal_front)
        normal_back = self.all_gather(normal_back)

        if len(rgb_pred.shape) > 4 and self.global_rank == 0:
            # Only do this for DDP training
            rgb_pred = rgb_pred.transpose(0, 1)    # batch_size x world_size x ...
            normal_pred = normal_pred.transpose(0, 1)    # batch_size x world_size x ...
            normal_front = normal_front.transpose(0, 1)    # batch_size x world_size x ...
            normal_back = normal_back.transpose(0, 1)    # batch_size x world_size x ...
            if self.val_size is not None:
                # Dirty way to handle the case when the # of GPUs does not evenly divide the size of validation dataset
                rgb_pred = rgb_pred.reshape(-1, *rgb_pred.shape[2:])[:self.val_size, ...]
                normal_pred = normal_pred.reshape(-1, *normal_pred.shape[2:])[:self.val_size, ...]
                normal_front = normal_front.reshape(-1, *normal_front.shape[2:])[:self.val_size, ...]
                normal_back = normal_back.reshape(-1, *normal_back.shape[2:])[:self.val_size, ...]
            else:
                rgb_pred = rgb_pred.reshape(-1, *rgb_pred.shape[2:])
                normal_pred = normal_pred.reshape(-1, *normal_pred.shape[2:])
                normal_front = normal_front.reshape(-1, *normal_front.shape[2:])
                normal_back = normal_back.reshape(-1, *normal_back.shape[2:])

        vis_dir = os.path.join(self.cfg['training']['out_dir'], 'vis')
        if os.path.exists(vis_dir):
            shutil.rmtree(vis_dir)

        os.makedirs(vis_dir)

        vid = []
        for idx, (rgb, normal, front, back) in enumerate(zip(rgb_pred, normal_pred, normal_front, normal_back)):
            rgb = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            normal = (normal.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            front = (front.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            back = (back.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)

            imageio.imwrite(os.path.join(vis_dir, "rgb_{:06d}.png".format(idx)), rgb)
            imageio.imwrite(os.path.join(vis_dir, "normal_{:06d}.png".format(idx)), normal)
            imageio.imwrite(os.path.join(vis_dir, "front_{:06d}.png".format(idx)), front)
            imageio.imwrite(os.path.join(vis_dir, "back_{:06d}.png".format(idx)), back)

            frame = np.concatenate([rgb, normal, front, back], axis=1)
            vid.append(frame)

        imageio.mimwrite(os.path.join(vis_dir, 'vis.mp4'), vid, fps=20)

    def configure_optimizers(self):
        # Intialize optimizer
        lr = self.cfg['training']['lr']
        pose_net_factor = self.cfg['training']['pose_net_factor']
        param_list = [
            {
                "params": self.model.sdf_decoder.net.layers.parameters(),
                "lr": lr,
            },
            {
                "params": self.model.sdf_decoder.pose_encoder.parameters(),
                "lr": lr * pose_net_factor,
            },
            {
                "params": self.model.color_decoder.parameters(),
                "lr": 1e-4,
            },
            {
                "params": self.model.deviation_decoder.parameters(),
                "lr": 1e-4,
            },
        ]

        if self.cfg['training']['train_skinning_net']:
            param_list += [
                    {
                        "params": self.model.skinning_model.parameters(),
                        "lr": self.cfg['training']['skinning_lr'],
                    },
                ]

        if self.cfg['model']['train_cameras']:
            param_list += [
                    {
                        "params": self.model.camera_parameters(),
                        "lr": 1e-4,
                    },
                ]

        if self.cfg['model']['train_smpl']:
            param_list += [
                    {
                        "params": self.model.smpl_parameters(),
                        "lr": 1e-4,
                    },
                ]

        if self.cfg['model']['color_pose_encoder'] in ['hybrid', 'latent'] or self.cfg['model']['geo_pose_encoder'] in ['latent']:
            param_list += [
                    {
                        "params": self.model.latent.parameters(),
                        "lr": 1e-4,
                        "weight_decay": 0.05,
                    },
                ]

        optimizer = torch.optim.Adam(params=param_list)

        return optimizer

    def compose_inputs(self, data, eval):
        cam_intri = data.get('image.K')

        skinning_weights = data.get('image.skinning_weights')
        smpl_verts = data.get('image.smpl_vertices')

        fg_mask = data.get('inputs.mask_erode')
        body_bounds_intersections = data.get('inputs.body_bounds_intersections')

        cam_idx = data.get('inputs.cam_idx')
        if self.model.train_cameras and not eval:
            # Optimize camera extrinsics
            uv = data.get('inputs.uv')

            cam_rot = conversions.quaternion_to_rotation_matrix(self.model.cam_rots[cam_idx], order=conversions.QuaternionCoeffOrder.XYZW)
            cam_trans = self.model.cam_trans[cam_idx]

            ray_dirs = get_camera_rays(cam_rot, uv)
            cam_loc = get_camera_location(cam_rot, cam_trans)
        else:
            # Use provided camera extrinsics
            ray_dirs = data.get('inputs.ray_dirs')
            cam_loc = data.get('image.cam_loc')

            cam_rot = data.get('image.R')
            cam_trans = data.get('image.T')

        batch_size = smpl_verts.size(0)

        coord_min = data.get('image.coord_min').view(batch_size, 1, -1)
        coord_max = data.get('image.coord_max').view(batch_size, 1, -1)
        center = data.get('image.center').view(batch_size, 1, -1)

        f_idx = data.get('inputs.frame_idx')[0].item()
        novel_seq = data.get('inputs.novel_seq')
        if novel_seq is not None:
            f_idx = -1

        if self.model.train_smpl and f_idx in self.model.frames:
            # Optimize estimated SMPL parameters
            # Get current SMPL parameters from the model
            root_orient = self.model.body_poses['root_orient_' + str(f_idx)].unsqueeze(0)

            pose_body = self.model.body_poses['pose_body_' + str(f_idx)].unsqueeze(0)
            pose_hand = self.model.body_poses['pose_hand_' + str(f_idx)].unsqueeze(0)
            trans = self.model.body_poses['trans_' + str(f_idx)].unsqueeze(0)

            betas = self.model.betas #.repeat(batch_size, 1)

            verts_posed, Jtrs, Jtrs_posed, bone_transforms, minimal_shape = self.model.forward_smpl(betas, root_orient, pose_body, pose_hand, trans)
            smpl_verts = (verts_posed + trans.unsqueeze(1)).repeat(batch_size, 1, 1)

            bone_transforms_02v = get_transforms_02v(Jtrs.squeeze(0))   # T-pose to Vitruvian A-pose

            # Put the canonical minimal shape mesh to Vitruvian A-pose
            T = torch.matmul(self.model.lbs_weights.clone(), bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
            minimal_shape_v = torch.matmul(T[:, :3, :3], minimal_shape.reshape(-1, 3, 1)).squeeze(-1) + T[:, :3, -1]

            center = torch.mean(minimal_shape_v, dim=0, keepdim=True)
            minimal_shape_v_centered = minimal_shape_v - center
            center = center.view(1, 1, -1).repeat(batch_size, 1, 1)

            coord_max = minimal_shape_v_centered.max().view(1, 1, -1).repeat(batch_size, 1, 1)
            coord_min = minimal_shape_v_centered.min().view(1, 1, -1).repeat(batch_size, 1, 1)
            minimal_shape = minimal_shape_v.reshape(1, -1, 3).repeat(batch_size, 1, 1)

            bone_transforms_02v = bone_transforms_02v.unsqueeze(0).repeat(batch_size ,1, 1, 1)
            bone_transforms = bone_transforms.repeat(batch_size ,1, 1, 1)

            Jtrs = normalize_canonical_points(Jtrs, coord_min[:1], coord_max[:1], center[:1])
            Jtrs = Jtrs.repeat(batch_size, 1, 1)

            Jtrs_posed = (Jtrs_posed + trans.unsqueeze(1))

            # Compute rots and rots_full
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1).reshape(-1, 3)   # (24, 3), poses in axis-angle representation
            full_pose_mat = conversions.angle_axis_to_rotation_matrix(full_pose).reshape(-1, 9)    # axis-angle to rotation matrix (24, 9)
            local_pose_mat = torch.cat([torch.eye(3, device=self.device, dtype=torch.float32).reshape(1, 9),
                                        full_pose_mat[1:, ...]], dim=0).reshape(-1, 9)   # 24 x 9, root rotation is set to identity

            # Expand to batch_size x 24 x 9
            rots = local_pose_mat.unsqueeze(0).repeat(batch_size, 1, 1)
            rots_full = full_pose_mat.unsqueeze(0)
        else:
            # Use provided SMPL parameters
            minimal_shape = data.get('image.minimal_shape')
            rots = data.get('image.rots')
            Jtrs = data.get('image.Jtrs')

            # TODO: this can only handle one single SMPL mesh at a time
            rots_full = data.get('image.rots_full')
            Jtrs_posed = data.get('image.Jtrs_posed')

            coord_min = data.get('image.coord_min').view(batch_size, 1, -1)
            coord_max = data.get('image.coord_max').view(batch_size, 1, -1)
            center = data.get('image.center').view(batch_size, 1, -1)

            bone_transforms = data.get('image.bone_transforms')
            bone_transforms_02v = data.get('image.bone_transforms_02v')
            trans = data.get('image.trans').unsqueeze(1)

        bone_transforms = torch.matmul(bone_transforms, torch.inverse(bone_transforms_02v)) # final bone transforms that transforms the canonical
                                                                                            # Vitruvian-pose mesh to the posed mesh, without global
                                                                                            # translation

        pose = torch.cat([F.pad(cam_rot, pad=(0, 0, 0, 1)),
                          F.pad(cam_trans, pad=(0, 1), value=1).unsqueeze(-1)],
                         dim=-1)    # camera pose

        pose_cond = {'rots_full': rots_full, 'Jtrs_posed': Jtrs_posed}
        if self.model.train_latent_code:
            if f_idx in self.model.frames:
                d_idx = data.get('inputs.data_idx')[:1]
            else:
                d_idx = torch.tensor([self.model.latent.num_embeddings - 1], dtype=torch.int64, device=self.device)

            pose_cond.update({'latent_code_idx': d_idx})

        inputs = {'intrinsics': cam_intri,
                  'ray_dirs': ray_dirs,
                  'body_bounds_intersections': body_bounds_intersections,
                  'cam_loc': cam_loc,
                  'cam_rot': cam_rot,
                  'cam_trans': cam_trans,
                  'pose': pose,
                  'body_mask': fg_mask,
                  'smpl_verts': smpl_verts,
                  'skinning_weights': skinning_weights,
                  'bone_transforms': bone_transforms,
                  'trans': trans,
                  'coord_min': coord_min,
                  'coord_max': coord_max,
                  'center': center,
                  'minimal_shape': minimal_shape,
                  'pose_cond': pose_cond,
                  'Jtrs': Jtrs,
                  'rots': rots,
                  'cam_idx': cam_idx,
                 }

        if self.model.train_geo_latent_code:
            if f_idx in self.model.frames:
                d_idx = data.get('inputs.data_idx')[:1]
            else:
                d_idx = torch.tensor([self.model.latent.num_embeddings - 1], dtype=torch.int64, device=self.device)

            inputs.update({'geo_latent_code_idx': d_idx})

        if not eval:
            rgb_values = data.get('inputs')
            inputs['rgb_values'] = rgb_values

            sampled_weights = data.get('image.sampled_weights')
            if sampled_weights is not None:
                inputs['sampled_weights'] = sampled_weights

            points_skinning = data.get('image.points_skinning')
            if points_skinning is not None:
                inputs['points_skinning'] = points_skinning

            points_inside = data.get('image.points_inside')
            if points_inside is not None:
                inputs['points_inside'] = points_inside

            points_uniform = data.get('image.points_uniform')
            if points_uniform is not None:
                inputs['points_uniform'] = points_uniform
        else:
            inputs['image_mask'] = data.get('inputs.image_mask')
            inputs['ray_dirs_cam'] = data.get('inputs.ray_dirs_cam')

        return inputs

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''

        inputs = self.compose_inputs(data, eval=False)

        model_outputs = self.model(inputs)

        ground_truth = {'rgb': inputs['rgb_values']}
        if 'sampled_weights' in inputs.keys():
            ground_truth.update({'sampled_weights': inputs['sampled_weights']})

        loss_dict = self.criteria(model_outputs, ground_truth)

        return loss_dict
