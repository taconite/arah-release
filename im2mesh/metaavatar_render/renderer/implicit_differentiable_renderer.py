import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from im2mesh.utils.diff_operators import gradient, jacobian

from im2mesh.utils.root_finding_utils import (
    forward_skinning, query_weights,
    unnormalize_canonical_points,
)


class IDHRNetwork(nn.Module):
    ''' Implicit Differentiable Human Renderer (IDHR) class.
    '''
    def __init__(self, deviation_network, rendering_network, skinning_model, ray_tracer, cano_view_dirs=True, train_skinning_net=False, render_last_pt=False, low_vram=False):
        ''' Initialization of the IDHRNetwork class.

        Args:
            deviation_decoder (torch.nn.Module): learnable scalar (i.e. b for volume rendering)
            rendering_network (torch.nn.Module): color network
            skinning_model (torch.nn.Module): skinning network
            ray_tracer (torch.nn.Module): ray tracer for sphere tracing (im2mesh.metaavatar_render.renderer.BodyRayTracing)
            cano_view_dirs (bool): whether to canonicalize viewing directions or not before feeding them to the color network
            train_skinning_net (bool): whether to optimize skinning network (with implicit gradients) or not
            render_last_pt (bool): if set to True, distance of the last point on ray will be set to 1e10, forcing the volume renderer
                                   to render the last point, regardless of its density
        '''
        super().__init__()
        self.deviation_network = deviation_network
        self.rendering_network = rendering_network
        self.skinning_model = skinning_model
        self.ray_tracer = ray_tracer

        self.cano_view_dirs = cano_view_dirs
        self.train_skinning_net = train_skinning_net
        self.render_last_pt = render_last_pt
        self.low_vram = low_vram

    def forward(self, input):
        ''' Forward pass of the model.

        Args:
            input (dict): dictionary containing input Tensors, etc.
        '''

        # Parse model input
        ## camera related inputs
        # intrinsics = input["intrinsics"]
        ray_dirs = input["ray_dirs"]
        cam_loc = input["cam_loc"]
        pose = input["pose"]

        ## human body related inputs
        body_mask = input["body_mask"]
        body_bounds_intersections = input['body_bounds_intersections']
        loc = input['loc']
        sc_factor = input['sc_factor']
        smpl_verts = input["smpl_verts"]
        skinning_weights = input["skinning_weights"]
        vol_feat = input["vol_feat"]
        bone_transforms = input["bone_transforms"]
        trans = input["trans"]
        coord_min = input["coord_min"]
        coord_max = input["coord_max"]
        center = input["center"]
        minimal_shape = input["minimal_shape"]
        sdf_network = input["sdf_network"]
        pose_cond = input["pose_cond"]

        if self.training:
            points_uniform = input["points_uniform"].reshape(-1, 3)
            if 'points_skinning' in input.keys():
                points_skinning = input["points_skinning"]
                pred_weights = query_weights(points_skinning, loc, sc_factor, coord_min, coord_max, center, self.skinning_model, vol_feat)
            else:
                pred_weights = None

        batch_size, num_pixels, _ = ray_dirs.shape
        device = ray_dirs.device

        sdf_network.eval()

        # Sphere tracing to find surface points and sample near/far-surface points
        with torch.no_grad():
            points_hat_norm, network_body_mask, dists, sampled_pts, sampled_dists, sampled_transforms, sampler_converge_mask \
                    = self.ray_tracer(sdf_network,
                                      self.skinning_model,
                                      cam_loc=cam_loc,
                                      ray_directions=ray_dirs,
                                      body_bounds_intersections=body_bounds_intersections,
                                      loc=loc,
                                      sc_factor=sc_factor,
                                      smpl_verts=smpl_verts,
                                      smpl_verts_cano=minimal_shape,
                                      skinning_weights=skinning_weights,
                                      vol_feat=vol_feat,
                                      bone_transforms=bone_transforms,
                                      trans=trans,
                                      coord_min=coord_min,
                                      coord_max=coord_max,
                                      center=center,
                                      eval_mode=not self.training,
                                     )

            if isinstance(sampled_dists, tuple):
                # sampled_bg_dists = sampled_dists[1]
                sampled_dists = sampled_dists[0]

        sdf_network.train()

        points_bar = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs) - trans
        points_cam = torch.matmul(points_bar + trans, pose[:, :3, :3].transpose(1, 2)) + pose[:, :3, -1].unsqueeze(1)

        if self.training:
            # Sample additional points in canonical space, to compute regularization losses
            if 'points_inside' in input.keys():
                points_inside = input["points_inside"]
                inside_sdf = sdf_network(points_inside).squeeze(0)
            else:
                inside_sdf = None

            # Sample points for the eikonal loss
            n_eik_points = 1024
            eikonal_points = (torch.rand(batch_size, n_eik_points, 3, device=device, dtype=torch.float32) - 0.5) * 2

            eikonal_points = eikonal_points.reshape(-1, 3)
            points_all = torch.cat([eikonal_points, points_uniform], dim=0)
            n_uniform_points = 1024
            n_uniform_points_total = batch_size * n_uniform_points
            n_eik_points_total = batch_size * n_eik_points

            # SDF and gradients
            points_all.requires_grad_(True)
            sdf_output = sdf_network(points_all).squeeze(0)
            uniform_sdf = sdf_output[n_eik_points_total:n_eik_points_total+n_uniform_points_total, :].reshape(batch_size, n_uniform_points, 1) # [inter_mask]
            grad_sdf = gradient(sdf_output, points_all)
            grad_eik = grad_sdf[:n_eik_points_total, :]
        else:
            valid_mask = (points_hat_norm.abs() <= 1.0).all(-1)
            surface_mask = network_body_mask & valid_mask

        rgb_values = torch.zeros_like(points_hat_norm)

        # Conduct SDF-based volume rendering
        vol_mask = sampler_converge_mask.any(-1)   # we render any ray that has at least 1 valid point
        ray_augm = False
        if self.training:
            ray_dirs_orig = ray_dirs.clone()
            if 'view_noise' in pose_cond.keys():
                # print ('use view noise')
                view_noise = pose_cond['view_noise']
                if view_noise is not None:
                    if view_noise.size(-1) == 3 and view_noise.size(-2) == 3:
                        ray_dirs = torch.matmul(view_noise, ray_dirs.transpose(1, 2)).transpose(1, 2)
                        ray_augm = True
                    else:
                        ray_dirs = ray_dirs + view_noise
                else:
                    ray_dirs = torch.zeros_like(ray_dirs)

            rgb_values[vol_mask], ws = self.get_rbg_value_vol_sdf(sdf_network,
                                                                  sampled_pts[vol_mask],
                                                                  sampled_dists[vol_mask],
                                                                  sampled_transforms[vol_mask],
                                                                  sampler_converge_mask[vol_mask],
                                                                  ray_dirs[vol_mask],
                                                                  ray_dirs_orig[vol_mask],
                                                                  pose_cond,
                                                                  loc[:1],
                                                                  sc_factor[:1],
                                                                  vol_feat,
                                                                  bone_transforms[:1],
                                                                  coord_min[:1],
                                                                  coord_max[:1],
                                                                  center[:1],
                                                                  ray_augm=ray_augm)
        else:
            sampled_pts = sampled_pts[vol_mask]
            sampled_dists = sampled_dists[vol_mask]
            sampled_transforms = sampled_transforms[vol_mask]
            sampler_converge_mask = sampler_converge_mask[vol_mask]
            ray_dirs = ray_dirs[vol_mask]

            if self.low_vram:
                p_split = torch.split(sampled_pts, 2048, dim=0)
                d_split = torch.split(sampled_dists, 2048, dim=0)
                t_split = torch.split(sampled_transforms, 2048, dim=0)
                m_split = torch.split(sampler_converge_mask, 2048, dim=0)
                r_split = torch.split(ray_dirs, 2048, dim=0)
            else:
                p_split = torch.split(sampled_pts, 20480, dim=0)
                d_split = torch.split(sampled_dists, 20480, dim=0)
                t_split = torch.split(sampled_transforms, 20480, dim=0)
                m_split = torch.split(sampler_converge_mask, 20480, dim=0)
                r_split = torch.split(ray_dirs, 20480, dim=0)

            rgb_values_vol = []
            ws_vol = []

            for pi, di, ti, mi,ri in zip(p_split, d_split, t_split, m_split, r_split):
                rgb_i, w_i = self.get_rbg_value_vol_sdf(sdf_network,
                                                        pi,
                                                        di,
                                                        ti,
                                                        mi,
                                                        ri,
                                                        ri.clone(),
                                                        pose_cond,
                                                        loc[:1],
                                                        sc_factor[:1],
                                                        vol_feat,
                                                        bone_transforms[:1],
                                                        coord_min[:1],
                                                        coord_max[:1],
                                                        center[:1],
                                                        ray_augm=ray_augm,
                                                        point_batch_size=100000 if self.low_vram else 1000000)

                rgb_values_vol.append(rgb_i)
                ws_vol.append(w_i)

            rgb_values[vol_mask] = torch.cat(rgb_values_vol, dim=0)
            ws = torch.cat(ws_vol, dim=0)

        network_body_mask = vol_mask
        mask_sdf = torch.zeros(batch_size, num_pixels, dtype=torch.float32, device=device)
        mask_sdf.masked_scatter_(vol_mask, ws)

        surface_normals = None

        if self.training:
            output = {
                'rgb_values': rgb_values,
                'sdf_output': mask_sdf,
                'network_body_mask': network_body_mask,
                'body_mask': body_mask,
                'off_surface_mask': vol_mask,
                'off_surface_sdf': uniform_sdf,
                'grad_theta': grad_eik,
                'surface_normals': surface_normals,
            }
            if pred_weights is not None:
                output.update({'pred_weights': pred_weights})
            if inside_sdf is not None:
                output.update({'inside_sdf': inside_sdf})

        else:
            points_cam[~surface_mask] = 0
            output = {
                # 'points': points_hat_norm,
                'points_cam': points_cam,
                'network_body_mask': network_body_mask,
                'rgb_values': rgb_values,
            }

        return output

    def get_rbg_value_vol_sdf(self,
                              sdf_network,
                              points,
                              z_vals,
                              transforms_fwd,
                              converge_mask,
                              view_dirs,
                              view_dirs_orig,
                              pose_cond,
                              loc,
                              sc_factor,
                              vol_feat,
                              bone_transforms,
                              coord_min,
                              coord_max,
                              center,
                              point_batch_size=100000,
                              ray_augm=False):

        n_pts, n_samples, _ = points.size()

        device = points.device

        lengths = converge_mask.sum(-1)
        pv = torch.arange(n_pts).to(device)
        scatter_mask= torch.zeros(n_pts, n_samples, device=device, dtype=bool)
        scatter_mask[pv, lengths - 1] = 1
        scatter_mask = scatter_mask +  torch.sum(scatter_mask, dim=1, keepdims=True) - torch.cumsum(scatter_mask, dim=1)
        scatter_mask = scatter_mask.bool()

        valid_points = points[converge_mask]
        valid_view_dirs = view_dirs.view(n_pts, 1, 3).repeat(1, n_samples, 1)[converge_mask]
        valid_view_dirs_orig = view_dirs_orig.view(n_pts, 1, 3).repeat(1, n_samples, 1)[converge_mask]
        valid_transforms_fwd = transforms_fwd[converge_mask]
        if self.cano_view_dirs:
            valid_transforms_bwd = valid_transforms_fwd.inverse().detach()
            input_view_dirs = torch.matmul(valid_transforms_bwd[:, :3, :3], -valid_view_dirs.unsqueeze(-1)).squeeze(-1).view(-1, 3)
            input_view_dirs_orig = torch.matmul(valid_transforms_bwd[:, :3, :3], -valid_view_dirs_orig.unsqueeze(-1)).squeeze(-1).view(-1, 3)
        else:
            input_view_dirs = -valid_view_dirs
            input_view_dirs_orig = -valid_view_dirs_orig

        p_split = torch.split(valid_points, point_batch_size, dim=0)
        v_split = torch.split(input_view_dirs, point_batch_size, dim=0)
        v_orig_split = torch.split(input_view_dirs_orig, point_batch_size, dim=0)
        t_split = torch.split(valid_transforms_fwd, point_batch_size, dim=0)

        # Get color and SDF predictions
        valid_sdf = []
        valid_rgb = []
        for idx, (pi, vi, vi_orig, ti) in enumerate(zip(p_split, v_split, v_orig_split, t_split)):
            with torch.enable_grad():
                pi = pi.unsqueeze(0).requires_grad_(True)

                if self.training and self.train_skinning_net:
                    # LBS and Jacobians
                    points_hat = unnormalize_canonical_points(pi, coord_min, coord_max, center)
                    points_lbs, _ = forward_skinning(points_hat,
                                                     loc=loc,
                                                     sc_factor=sc_factor,
                                                     coord_min=coord_min,
                                                     coord_max=coord_max,
                                                     center=center,
                                                     skinning_model=self.skinning_model,
                                                     vol_feat=vol_feat,
                                                     bone_transforms=bone_transforms,
                                                     mask=None,
                                                     point_batch_size=point_batch_size
                                                    )

                    jacobian_lbs, status = jacobian(points_lbs, pi)
                    jacobian_lbs_inv = jacobian_lbs.inverse().detach()

                    pi = pi - torch.matmul( jacobian_lbs_inv, (points_lbs - points_lbs.clone().detach()).unsqueeze(-1) ).squeeze(-1)

                feature_vectors = sdf_network[:-1](pi).squeeze(0)
                sdf = sdf_network[-1](feature_vectors)
                normal = gradient(sdf, pi) if self.training else gradient(sdf, pi, create_graph=False)
                if not self.cano_view_dirs:
                    normal = torch.einsum('pij,bpj->bpi', ti[:, :3, :3], normal)

                if self.training and ray_augm:
                    with torch.no_grad():
                        normal_n = normal / torch.linalg.norm(normal, ord=2, dim=-1, keepdim=True)
                        nv_dots = (normal_n.squeeze(0) * vi).sum(-1)
                        angles = torch.arccos(nv_dots)
                        invalid_mask = angles >= np.pi / 2.0

                    # print ('# of invalid points: {}/{}'.format(invalid_mask.sum().item(), invalid_mask.numel()))
                    vi[invalid_mask] = vi_orig[invalid_mask]

            if not self.training:
                # Make sure no backward graph is stored at test time
                normal = normal.detach()
                feature_vectors = feature_vectors.detach()
                sdf = sdf.detach()
                pi = pi.detach()

            sdf = sdf / 2.0 * 1.1 * (coord_max.squeeze() - coord_min.squeeze())
            valid_sdf.append(sdf.squeeze(0))
            valid_rgb.append(self.rendering_network(pi.squeeze(0), normal.squeeze(0), vi, feature_vectors, pose_cond))

        valid_sdf = torch.cat(valid_sdf, dim=0)
        valid_rgb = torch.cat(valid_rgb, dim=0)

        beta = self.deviation_network(valid_sdf).clip(1e-6, 1e6)
        inv_beta = torch.reciprocal(beta)
        valid_density = F.relu(inv_beta * (0.5 + 0.5 * torch.sign(-valid_sdf) * (1 - torch.exp(-torch.abs(-valid_sdf) * inv_beta))))

        rgb_vals = torch.zeros(n_pts, n_samples, 3, dtype=torch.float32, device=device)
        density = torch.zeros(n_pts, n_samples, dtype=torch.float32, device=device)
        new_z_vals = 1e10 * torch.ones(n_pts, n_samples, dtype=torch.float32, device=device)

        rgb_vals.masked_scatter_(scatter_mask.unsqueeze(-1), valid_rgb)
        density.masked_scatter_(scatter_mask, valid_density)
        new_z_vals.masked_scatter_(scatter_mask, z_vals[converge_mask])

        # Volume rendering with VolSDF formulation
        dists = new_z_vals[...,1:] - new_z_vals[...,:-1]
        if self.render_last_pt:
            dists = torch.cat([dists, 1e10 * torch.ones(n_pts, 1, device=device, dtype=torch.float32)], dim=-1)  # [N_rays, N_samples]
        else:
            dists = torch.cat([dists, 1. / self.ray_tracer.n_steps * torch.ones(n_pts, 1, device=device, dtype=torch.float32)], dim=-1)  # [N_rays, N_samples]
            pv = torch.arange(n_pts).to(device)
            dists[pv, (lengths - 1)] = 1. / self.ray_tracer.n_steps

        alpha = 1.0 - torch.exp(-density*dists)

        weights = alpha * \
                torch.cumprod(torch.cat([torch.ones(n_pts, 1, dtype=torch.float32, device=device), 1. - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]

        weights_sum = (weights * scatter_mask).sum(dim=-1, keepdim=True).clip(0, 1)

        rgb_vals = ((rgb_vals * weights.unsqueeze(-1)) * scatter_mask.unsqueeze(-1)).sum(dim=1)

        return rgb_vals, weights_sum
