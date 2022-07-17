import torch
import lpips
from torch import nn
from torch.nn import functional as F

class IDHRLoss(nn.Module):
    ''' Loss class for Implicit Differentiable Human Renderer (IDHR) '''

    def __init__(self, rgb_weight, perceptual_weight, eikonal_weight, mask_weight, off_surface_weight, inside_weight, params_weight, skinning_weight, rgb_loss_type='l1', perceptual_loss_fn=None):
        ''' Initialization of the trainer class.

        Args:
            rgb_weight (float): weight fo RGB loss
            perceptual_weight (float): weight for perceptual loss
            eikonal_weight (float): weight for eikonal loss
            mask_weight (float): weight for mask loss
            off_surface_weight (float): weight for off-surface point loss
            inside_weight (float): weight for inside loss
            params_weight (float): weight for SDF parameter regularization loss
            skinning_weight (float): weight for skinning loss
            rgb_loss_type (str): type of the RGB loss (either L1, L2, or smoothed L1)
            perceptual_loss_fn (torch.nn.Module): differentiable function which computes perceptual losses (e.g. LPIPS)
        '''

        super().__init__()
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.off_surface_weight = off_surface_weight
        self.params_weight = params_weight
        self.skinning_weight = skinning_weight
        self.inside_weight = inside_weight
        # self.alpha = alpha
        if rgb_loss_type == 'l1':
            self.l1_loss = nn.L1Loss(reduction='sum')
        elif rgb_loss_type == 'mse':
            self.l1_loss = nn.MSELoss(reduction='sum')
        elif rgb_loss_type == 'smoothed_l1':
            self.l1_loss = nn.SmoothL1Loss(reduction='sum', beta=1e-1)
        else:
            raise ValueError('Unsupported RGB loss type: {}. Only l1, smoothed_l1 and mse are supported'.format(rgb_loss_type))

        self.p_loss = perceptual_loss_fn

    def get_rgb_loss(self, rgb_values, rgb_gt, network_body_mask, body_mask):
        device = body_mask.device

        if network_body_mask.sum() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        if body_mask.max() > 1:
            # Ignore boundary pixels if we use patch sampling
            network_body_mask = network_body_mask & (body_mask != 100)

        rgb_values = rgb_values[network_body_mask]
        rgb_gt = rgb_gt[network_body_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(network_body_mask.numel())
        # rgb_loss = torch.norm(rgb_values - rgb_gt, p=1, dim=-1).sum() / float(body_mask.numel())
        return rgb_loss

    def get_perceptual_loss(self, rgb_values, rgb_gt, network_body_mask):
        device = network_body_mask.device
        if network_body_mask.sum() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        # TODO: this code is ugly, it only works for hybrid sampling
        # Reshape, permute and transform to [-1, 1]
        # pred_patch = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)(rgb_values[:, 1024:, :].reshape(-1, 32, 32, 3).permute(0, 3, 1, 2))
        # gt_patch = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)(rgb_gt[:, 1024:, :].reshape(-1, 32, 32, 3).permute(0, 3, 1, 2))
        pred_patch = rgb_values.reshape(-1, 48, 48, 3).permute(0, 3, 1, 2)
        gt_patch = rgb_gt.reshape(-1, 48, 48, 3).permute(0, 3, 1, 2)

        # import cv2
        # import numpy as np
        # patch_img = (pred_patch.permute(0, 2, 3, 1).view(2, 32, 32, 3)[0].detach().cpu().numpy() * 255).astype(np.uint8)
        # gt_img = (gt_patch.permute(0, 2, 3, 1).view(2, 32, 32, 3)[0].detach().cpu().numpy() * 255).astype(np.uint8)
        # cv2.imwrite('tmp/patch.png', cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('tmp/gt.png', cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        # import pdb
        # pdb.set_trace()

        perceptual_loss = self.p_loss(pred_patch, gt_patch, normalize=True).mean()
        return perceptual_loss

    def get_eikonal_loss(self, grad_theta, body_mask):
        device = body_mask.device
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        eikonal_loss = torch.abs((grad_theta.norm(2, dim=-1) - 1)).sum() / float(body_mask.numel())
        return eikonal_loss

    def get_mask_loss_vol_sdf(self, weights_output, body_mask, off_surface_mask):
        device = body_mask.device
        if off_surface_mask.sum() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        gt = body_mask[off_surface_mask].float()
        # mask_loss = F.binary_cross_entropy(weights_output[off_surface_mask], gt, reduction='none').sum() / float(body_mask.numel())
        mask_loss = torch.norm(weights_output[off_surface_mask] - gt, dim=-1).sum() / float(body_mask.numel())
        return mask_loss

    def get_off_surface_loss(self, off_surface_sdf, body_mask):
        # return torch.exp(-1e2 * torch.abs(off_surface_sdf)).sum() / float(body_mask.numel())
        return torch.exp(-1e2 * off_surface_sdf).sum() / float(body_mask.numel())

    def get_sdf_params_loss(self, sdf_params):
        sdf_params = torch.cat(sdf_params, dim=1)
        n_params = sdf_params.size(-1)

        return sdf_params.norm(dim=-1).mean() / n_params

    def get_normals_loss(self, normals, body_mask):
        return torch.norm(normals[0] - normals[1], dim=-1).sum() / float(body_mask.numel())

    def get_skinning_loss(self, pred, target):
        return torch.abs(pred - target).sum(-1).mean()

    def get_inside_loss(self, inside_sdf, body_mask):
        return torch.sigmoid(inside_sdf * 5e3).sum() / float(body_mask.numel())

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        network_body_mask = model_outputs['network_body_mask'][:, :2048]
        body_mask = model_outputs['body_mask'][:, :2048]
        off_surface_mask = model_outputs['off_surface_mask'][:, :2048]
        normals = model_outputs['surface_normals']

        device = model_outputs['rgb_values'].device

        if self.rgb_weight > 0:
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][:, :2048], rgb_gt[:, :2048], network_body_mask, body_mask)
        else:
            rgb_loss = torch.zeros(1, device=device)

        if self.perceptual_weight > 0:
            perceptual_loss = self.get_perceptual_loss(model_outputs['rgb_values'][:, 2048:], rgb_gt[:, 2048:], network_body_mask)
        else:
            perceptual_loss = torch.zeros(1, device=device)

        if self.mask_weight > 0:
            mask_loss = self.get_mask_loss_vol_sdf(model_outputs['sdf_output'], body_mask, off_surface_mask)
        else:
            mask_loss = torch.zeros(1, device=device)

        if self.eikonal_weight > 0:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'], body_mask)
        else:
            eikonal_loss = torch.zeros(1, device=device)

        if self.off_surface_weight > 0:
            off_surface_loss = self.get_off_surface_loss(model_outputs['off_surface_sdf'], body_mask)
        else:
            off_surface_loss = torch.zeros(1, device=device)

        if self.inside_weight > 0:
            inside_loss = self.get_inside_loss(model_outputs['inside_sdf'], body_mask)
        else:
            inside_loss = torch.zeros(1, device=device)

        if self.params_weight > 0:
            sdf_params_loss = self.get_sdf_params_loss(model_outputs['sdf_params'])
        else:
            sdf_params_loss = torch.zeros(1, device=device)

        if self.skinning_weight > 0:
            skinning_loss = self.get_skinning_loss(model_outputs['pred_weights'], ground_truth['sampled_weights'])
        else:
            skinning_loss = torch.zeros(1, device=device)

        loss = self.rgb_weight * rgb_loss + \
               self.perceptual_weight * perceptual_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.off_surface_weight * off_surface_loss + \
               self.inside_weight * inside_loss + \
               self.params_weight * sdf_params_loss + \
               self.skinning_weight * skinning_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'perceptual_loss': perceptual_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'off_surface_loss': off_surface_loss,
            'inside_loss': inside_loss,
            'sdf_params_loss': sdf_params_loss,
            'skinning_loss': skinning_loss,
        }
