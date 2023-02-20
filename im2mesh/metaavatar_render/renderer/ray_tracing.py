import torch
import pytorch3d
import torch.nn as nn
from collections import OrderedDict

from im2mesh.utils.root_finding_utils import (
    normalize_canonical_points, unnormalize_canonical_points,
    search_canonical_corr, search_iso_surface_depth,
    eval_sdf
)


class BodyRayTracing(nn.Module):
    ''' Ray-tracer for articulated human body SDF.
    '''
    def __init__(
            self,
            root_finding_threshold=1.0e-5,
            sphere_tracing_iters=50,
            n_steps=64,
            near_surface_vol_samples=16,
            far_surface_vol_samples=16,
            surface_vol_range=0.05,
            sample_bg_pts=0,
            low_vram=False,
    ):
        ''' Initialization of the ray-tracer class.

        Args:
            root_finding_threshold (float): error threshold for terminating root-finding
            sphre_tracing_iters (int): number of steps of sphere tracing to get rough estimations based on SMPL skinning weights
            n_steps (int): number of samples for rays that do not intersect with surface
            near_surface_vol_samples (int): number of near-surface samples for rays that intersect with surface
            far_surface_vol_samples (int): number of far-surface samples for rays that intersect with surface
            surface_vol_range (float): range from which to sample near-surface samples for rays that intersect with surface (in meters)
            sample_bg_pts (int): if greater than 0, sample points for background NeRF model (doesn't quite work)
        '''
        super().__init__()

        self.root_finding_threshold = root_finding_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.n_steps = n_steps

        self.near_surface_vol_samples = near_surface_vol_samples
        self.surface_vol_range = surface_vol_range
        self.far_surface_vol_samples = far_surface_vol_samples

        self.sample_bg_pts = sample_bg_pts
        self.low_vram = low_vram

    def forward(self,
                # networks
                sdf_network,
                skinning_model,
                # cameras
                cam_loc,
                ray_directions,
                body_bounds_intersections,
                # SMPL-related inputs
                loc,
                sc_factor,
                smpl_verts,
                smpl_verts_cano,
                skinning_weights,
                vol_feat,
                bone_transforms,
                trans,
                coord_min,
                coord_max,
                center,
                eval_mode=False
                ):
        ''' Forward pass of the ray-tracer.

        Args:
            sdf_network (torch.nn.Module): SDF network
            skinning_model (torch.nn.Module): skinning network
            cam_loc (torch.Tensor of batch_size x 3): camera centers in world coordinate
            ray_directions (torch.Tensor of batch_size x n_pixels x 3): camera ray directions for each pixel
            body_bounds_intersections (torch.Tensor of batch_size x n_pixels x 2): near and far bounds for each camera ray
            loc (torch.Tensor of batch_size x 1 x 3): center location of the canonical SMPL mesh (0 for default setting)
            sc_factor (torch.Tensor of batch_size x 1 x 3): scaling of the canonical SMPL mesh (1 for default setting)
            smpl_verts (torch.Tensor of batch_size x 6890 x 3): vertices of registered SMPL meshes in transformed space
            smpl_verts_cano (torch.Tensor of batch_size x 6890 x 3): vertices of canonical SMPL meshes
            skinning_weights (torch.Tensor of batch_size x 6890 x 3): skinning weights of SMPL vertices
            vol_feat (torch.Tensor or dictionary): volume feature for skinning network encoder. Not used by the default model
            bone_transforms (torch.Tensor of batch_size x 24 x 4 x 4): SMPL bone transforms that transform canonical SMPL mesh to posed SMPL mesh,
                                                                       without global translation
            trans (torch.Tensor of batch_size x 1 x 3): global translation of registered SMPL meshes
            coord_min (torch.Tensor of batch_size x 1 x 3): lower bounds for bounding boxes of the canonical SMPL meshes
            coord_max (torch.Tensor of batch_size x 1 x 3): upper bounds for bounding boxes of the canonical SMPL meshes
            center (torch.Tensor of batch_size x 1 x 3): centroids of the canonical SMPL meshes
            eval_mode (bool): use evaluation mode (no random perturbation for ray samples)
        '''

        batch_size, num_pixels, _ = ray_directions.shape

        if isinstance(vol_feat, OrderedDict):
            vol_feat_ = OrderedDict()
            for k, v in vol_feat.items():
                vol_feat_[k] = v[:1, ...]

            vol_feat = vol_feat_

        cam_locs = cam_loc.view(batch_size, 1, 3).repeat(1, num_pixels, 1)

        assert(self.near_surface_vol_samples > 0 or self.far_surface_vol_samples > 0)
        # Sphere tracing to find surface points
        curr_start_points, curr_start_transforms, unfinished_mask_start, acc_start_dis, acc_end_dis = \
            self.sphere_tracing(batch_size,
                                num_pixels,
                                sdf_network,
                                skinning_model,
                                cam_locs,
                                ray_directions,
                                # torch.ones(batch_size, num_pixels, device=body_mask.device, dtype=bool),
                                body_bounds_intersections,
                                loc[:1],
                                sc_factor[:1],
                                smpl_verts[:1],
                                smpl_verts_cano[:1],
                                skinning_weights[:1],
                                vol_feat,
                                bone_transforms[:1],
                                trans[:1],
                                coord_min[:1],
                                coord_max[:1],
                                center[:1],
                                eval_mode=eval_mode)    # Note we assume all batches come from different views of the same temporal frame,
                                                        # so SMPL-related parameters are the same across batches. Otherwise it is hard to
                                                        # call the SDF network consistently since the SDF network is generated from a
                                                        # hypernetwork which takes body poses as inputs

        # Sample for volume rendering: sample near_surface_vol_samples points around surface points;
        # sample far_surface_vol_samples points outside surface range;
        # sample n_steps points for non-convergent rays
        network_body_mask = (~unfinished_mask_start)    # converged rays from sphere tracing
        sampler_mask = torch.ones_like(unfinished_mask_start)   # run sampler on all rays

        if sampler_mask.sum() > 0:
            sampler_min_max = torch.stack([acc_start_dis, acc_end_dis], dim=-1)

            sampler_pts, sampler_transforms, sampler_converge_mask, sampler_dists = \
                    self.ray_sampler(sdf_network,
                                     skinning_model,
                                     cam_locs,
                                     network_body_mask,
                                     ray_directions,
                                     sampler_min_max,
                                     body_bounds_intersections,
                                     sampler_mask,
                                     loc[:1],
                                     sc_factor[:1],
                                     smpl_verts[:1],
                                     smpl_verts_cano[:1],
                                     skinning_weights[:1],
                                     vol_feat,
                                     bone_transforms[:1],
                                     trans[:1],
                                     coord_min[:1],
                                     coord_max[:1],
                                     center[:1],
                                     eval_mode
                                    )

        return curr_start_points, \
               network_body_mask, \
               acc_start_dis, \
               sampler_pts, \
               sampler_dists, \
               sampler_transforms, \
               sampler_converge_mask

    def sphere_tracing(self, batch_size, n_pts, sdf_network, skinning_model, cam_locs, ray_directions, body_bounds_intersections, loc, sc_factor, smpl_verts, smpl_verts_cano, skinning_weights, vol_feat, bone_transforms, trans, coord_min, coord_max, center, clamp_dist=0.1, eval_mode=False):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''
        device = cam_locs.device

        body_ray_directions = ray_directions.reshape(1, -1, 3).clone()
        acc_start_dis = body_bounds_intersections[..., 0].reshape(1, -1).clone()
        acc_end_dis = body_bounds_intersections[..., 1].reshape(1, -1).clone()

        assert (acc_start_dis <= acc_end_dis).all()

        cam_locs = cam_locs.reshape(1, -1, 3).clone()

        acc_start_dis_orig = acc_start_dis.clone()
        acc_end_dis_orig = acc_end_dis.clone()

        # For sphere-tracing, record which rays has not converged yet
        unfinished_mask_start = acc_start_dis < acc_end_dis

        # For sphere-tracing, record which rays has already diverged, i.e. exceeding the far bound
        diverge_mask = acc_start_dis >= acc_end_dis

        # Fast sphere-tracing via nearest-neighbor skinning to find approximate frontal surface points
        curr_start_points = curr_start_transforms = None

        for _ in range(self.sphere_tracing_iters):
            x_hat, sdf, transform_fwd, converge_mask = self.generate_point_samples_opt(sdf_network,
                                                                                       cam_locs,
                                                                                       body_ray_directions,
                                                                                       unfinished_mask_start,
                                                                                       acc_start_dis,
                                                                                       loc,
                                                                                       sc_factor,
                                                                                       smpl_verts,
                                                                                       skinning_weights,
                                                                                       skinning_model,
                                                                                       vol_feat,
                                                                                       bone_transforms,
                                                                                       trans,
                                                                                       coord_min,
                                                                                       coord_max,
                                                                                       center,
                                                                                       use_opt=False,
                                                                                      )



            if curr_start_points is None and curr_start_transforms is None:
                curr_start_points = x_hat
                curr_start_transforms = transform_fwd
            else:
                curr_start_points[converge_mask] = x_hat[converge_mask]
                curr_start_transforms[converge_mask] = transform_fwd[converge_mask]

            # clamp sdf
            sdf_marching = torch.clamp(sdf, -clamp_dist, clamp_dist)

            # ray marching
            update_mask = (sdf_marching.abs() > self.root_finding_threshold) & (sdf.abs() < 1e6)
            acc_start_dis[update_mask] = acc_start_dis[update_mask] + sdf_marching[update_mask]

            # find diverged rays
            diverge_mask[update_mask] = acc_start_dis[update_mask] >= acc_end_dis[update_mask]

            # remove rays that either converged or diverged
            remove_mask = (converge_mask & (sdf.abs() <= self.root_finding_threshold)) | diverge_mask
            # remove_mask = (converge_mask & (sdf <= self.root_finding_threshold * 2)) | diverge_mask

            unfinished_mask_start[remove_mask] = 0  # stop marching on already converged/diverged rays

        # For non-divergent (during testing, to save computation) or all (during training) rays,
        # run root-finding to find SDF iso-surface points and their depth on camera rays
        curr_start_points = unnormalize_canonical_points(curr_start_points, coord_min, coord_max, center)
        curr_start_points_opt, acc_start_dis_opt, curr_start_transforms_opt, converge_mask_opt = \
                search_iso_surface_depth(cam_locs,
                                         body_ray_directions,
                                         ~diverge_mask if eval_mode else torch.ones_like(diverge_mask),
                                         curr_start_points,
                                         acc_start_dis,
                                         curr_start_transforms,
                                         sdf_network,
                                         loc,
                                         sc_factor,
                                         skinning_model,
                                         vol_feat,
                                         bone_transforms,
                                         trans,
                                         coord_min,
                                         coord_max,
                                         center,
                                         eval_mode=True)

        # Only optimized depth points which are between near/far bounds are considered to be converged
        converge_mask_opt = converge_mask_opt & (acc_start_dis_opt >= acc_start_dis_orig) & (acc_start_dis_opt <= acc_end_dis_orig)

        curr_start_points_opt_norm = normalize_canonical_points(curr_start_points_opt, coord_min, coord_max, center)

        diverge_mask =  diverge_mask & ~converge_mask_opt   # sometime during training, root-finding can fix some diverged rays
        # Optimized depth points which are outside the sepecified depth range are considered as diverged
        diverge_mask =  diverge_mask | (acc_start_dis_opt < acc_start_dis_orig) | (acc_start_dis_opt > acc_end_dis_orig)

        acc_start_dis = acc_start_dis_opt.clone()

        # Reset acc_start_dis/acc_end_dis to min/max for non-convergent rays
        acc_start_dis[~converge_mask_opt] = acc_start_dis_orig[~converge_mask_opt].clone()
        acc_end_dis[~converge_mask_opt] = acc_end_dis_orig[~converge_mask_opt].clone()

        # All rays that do not converge will be used in ray sampler
        unfinished_mask_start = (~converge_mask_opt)

        # The following tensors are:
        # curr_start_points_ret - canonical points. Filled with root-finding (convergent rays) and knn forward marching (non-convergent rays)
        # curr_start_transforms_ret - forward transformations for canonical points
        # unfinished_mask_start_ret - mask for non-convergent rays. Non-convergent rays include divergent rays
        # acc_start_dis_ret - depth of forward ray-marching. Filled with root-finding solutions (convergent rays) and near scene bounds
        #                     (non-convergent rays)
        # acc_end_dis_ret - far bounds of ray sampling
        curr_start_points_ret = curr_start_points_opt_norm.reshape(batch_size, n_pts, 3).clone()
        curr_start_transforms_ret = curr_start_transforms_opt.reshape(batch_size, n_pts, 4, 4).clone()
        unfinished_mask_start_ret = unfinished_mask_start.reshape(batch_size, n_pts).clone()
        acc_start_dis_ret = acc_start_dis.reshape(batch_size, n_pts).clone()
        acc_end_dis_ret = acc_end_dis.reshape(batch_size, n_pts).clone()

        return curr_start_points_ret, curr_start_transforms_ret, unfinished_mask_start_ret, acc_start_dis_ret, acc_end_dis_ret

    def perturb_z_vals(self, z_vals, fix_idx=None):
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(upper)
        if fix_idx is not None:
            # Don't perturb surface points
            t_rand[..., fix_idx] = 0.5

        z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def ray_sampler(self, sdf_network, skinning_model, cam_locs, body_mask, ray_directions, sampler_min_max, bounds_min_max, sampler_mask, loc, sc_factor, smpl_verts, smpl_verts_cano, skinning_weights, vol_feat, bone_transforms, trans, coord_min, coord_max, center, eval_mode=False):
        ''' Sample on rays in given ranges '''
        # Sample n_samples points on non-convergent rays
        device = sampler_min_max.device
        z_vals = torch.linspace(0.0, 1.0, self.n_steps, device=device, dtype=torch.float32).view(1, 1, -1)
        z_vals = sampler_min_max[..., :1] + (sampler_min_max[..., 1:] - sampler_min_max[..., :1]) * z_vals # (batch_size, n_pts, n_samples)
        if not eval_mode:
            z_vals = self.perturb_z_vals(z_vals)

        batch_size, n_pts, n_samples = z_vals.size()
        sampler_mask = sampler_mask.view(batch_size, n_pts, 1).repeat(1, 1, n_samples)

        # For convergent rays, we sample (near_surface_vol_samples+1) points, including the surface points,
        # in the range [surface_depth - self.surface_vol_range, surface_depth + self.surface_vol_range]
        if self.near_surface_vol_samples > 0 or self.far_surface_vol_samples > 0:
            body_indices = torch.nonzero(body_mask)
            sampler_mask[body_indices[:, 0], body_indices[:, 1], self.near_surface_vol_samples+1:] = 0
            z_vals_surface = torch.linspace(0.0, 1.0, self.near_surface_vol_samples+1, device=device, dtype=torch.float32).view(1, 1, -1)
            surface_depth = sampler_min_max[..., :1]
            z_vals_surface = surface_depth - self.surface_vol_range + self.surface_vol_range * 2 * z_vals_surface # (batch_size, n_pts, self.near_surface_vol_samples+1)
            if not eval_mode:
                z_vals_surface = self.perturb_z_vals(z_vals_surface, fix_idx=self.near_surface_vol_samples // 2)

            z_vals[body_indices[:, 0], body_indices[:, 1], :self.near_surface_vol_samples+1] = z_vals_surface[body_indices[:, 0], body_indices[:, 1]]

            if self.far_surface_vol_samples > 0:
                sampler_mask[body_indices[:, 0], body_indices[:, 1], self.near_surface_vol_samples+1:self.near_surface_vol_samples+1+self.far_surface_vol_samples] = 1
                z_vals_far = torch.linspace(0.0, 1.0, self.far_surface_vol_samples, device=device, dtype=torch.float32).view(1, 1, -1)
                z_vals_far = bounds_min_max[..., :1] + torch.maximum(surface_depth - self.surface_vol_range - bounds_min_max[..., :1],
                                                                     torch.Tensor([1e-5]).float().to(device)) * z_vals_far # (batch_size, n_pts, self.far_surface_vol_samples)
                if not eval_mode:
                    z_vals_far = self.perturb_z_vals(z_vals_far)

                z_vals[body_indices[:, 0], body_indices[:, 1], self.near_surface_vol_samples+1:self.near_surface_vol_samples+1+self.far_surface_vol_samples] = z_vals_far[body_indices[:, 0], body_indices[:, 1]]

                sorted_z_vals, sorted_inds = torch.sort(z_vals[body_indices[:, 0], body_indices[:, 1], :self.near_surface_vol_samples+1+self.far_surface_vol_samples])

                z_vals[body_indices[:, 0], body_indices[:, 1], :self.near_surface_vol_samples+1+self.far_surface_vol_samples] = sorted_z_vals

        # Convert sampled points to canonical space
        sampler_pts, sampler_transforms, sampler_converge_mask \
                = self.generate_point_samples_opt(sdf_network,
                                                  cam_locs,
                                                  ray_directions,
                                                  sampler_mask,
                                                  z_vals,
                                                  loc,
                                                  sc_factor,
                                                  smpl_verts,
                                                  skinning_weights,
                                                  skinning_model,
                                                  vol_feat,
                                                  bone_transforms,
                                                  trans,
                                                  coord_min,
                                                  coord_max,
                                                  center,
                                                  use_opt=True,
                                                  return_sdf=False,
                                                  eval_mode=eval_mode)

        # Sample points for rendering background, if applicable
        if self.sample_bg_pts > 0 and not eval_mode:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.sample_bg_pts + 1.0), self.sample_bg_pts, device=device, dtype=torch.float32).view(1, 1, -1)
            z_vals_outside = bounds_min_max[..., 1:] / torch.flip(z_vals_outside, dims=[-1]) # + 1.0 / self.n_steps
            z_vals = (z_vals, z_vals_outside)

        return sampler_pts, sampler_transforms, sampler_converge_mask, z_vals

    def inv_transform_points_smpl_verts(self, points, smpl_verts, skinning_weights, bone_transforms, trans, coord_min, coord_max, center):
        ''' Backward skinning based on nearest neighbor SMPL skinning weights '''
        batch_size, n_pts, _ = points.size()
        device = points.device
        knn_ret = pytorch3d.ops.knn_points(points, smpl_verts)
        p_idx = knn_ret.idx.squeeze(-1)
        bv, _ = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(n_pts).to(device)], indexing='ij')
        pts_W = skinning_weights[bv, p_idx, :]
        # _, part_idx = pts_W.max(-1)

        transforms_fwd = torch.matmul(pts_W, bone_transforms.view(batch_size, -1, 16)).view(batch_size, n_pts, 4, 4)
        transforms_bwd = torch.inverse(transforms_fwd)

        homogen_coord = torch.ones(batch_size, n_pts, 1, dtype=torch.float32, device=device)
        points_homo = torch.cat([points - trans, homogen_coord], dim=-1).view(batch_size, n_pts, 4, 1)
        points_new = torch.matmul(transforms_bwd, points_homo)[:, :, :3, 0]
        points_new = normalize_canonical_points(points_new, coord_min, coord_max, center)

        return points_new, transforms_fwd


    def inv_transform_points_opt(self, x_bar, sdf_network, loc, sc_factor, smpl_verts, skinning_weights, skinning_model, vol_feat, bone_transforms, trans, coord_min, coord_max, center, eval_mode=False):
        ''' Backward skinning via forward LBS root-finding '''
        batch_size, n_pts, _ = x_bar.size()
        device = x_bar.device
        knn_ret = pytorch3d.ops.knn_points(x_bar, smpl_verts, K=1)
        init_bones = knn_ret.idx
        x_bar = x_bar - trans

        p_idx = init_bones.squeeze(-1)
        bv, _ = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(n_pts).to(device)], indexing='ij')
        pts_W = skinning_weights[bv, p_idx, :]

        T_fwd_0 = torch.matmul(pts_W, bone_transforms.view(batch_size, -1, 16)).view(batch_size, n_pts, 4, 4)
        T_bwd_0 = torch.inverse(T_fwd_0)

        homogen_coord = torch.ones(batch_size, n_pts, 1, dtype=torch.float32, device=device)
        points_homo = torch.cat([x_bar, homogen_coord], dim=-1).view(batch_size, n_pts, 4, 1)
        x_hat_0 = torch.matmul(T_bwd_0, points_homo)[:, :, :3, 0]
        x_hat_0 = x_hat_0.unsqueeze(2)  # (batch_size, n_pts, 1, 3)
        T_fwd_0 = T_fwd_0.unsqueeze(2)  # (batch_size, n_pts, 1, 3)

        # Search the canonical correspondence and return forward bone transforms
        if not eval_mode:
            eval_point_batch_size = 100000
        elif self.low_vram:
            eval_point_batch_size = 100000
        else:
            eval_point_batch_size = 1000000

        x_hat_opt, T_fwd_opt, converge_mask, error_opt = \
                search_canonical_corr(x_bar,
                                      x_hat_0,
                                      T_fwd_0,
                                      loc,
                                      sc_factor,
                                      skinning_model,
                                      vol_feat,
                                      bone_transforms,
                                      coord_min,
                                      coord_max,
                                      center,
                                      eval_mode=eval_mode,
                                      eval_point_batch_size=eval_point_batch_size)

        converge_inds = torch.zeros(batch_size, n_pts, device=device, dtype=torch.int64)

        bv, pv = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(n_pts).to(device)], indexing='ij')

        x_hat_opt = x_hat_opt[bv, pv, converge_inds]
        T_fwd_opt = T_fwd_opt[bv, pv, converge_inds]

        converge_mask = converge_mask.sum(-1) > 0

        x_hat = x_hat_opt
        T_fwd = T_fwd_opt

        x_hat_norm = normalize_canonical_points(x_hat, coord_min, coord_max, center)

        return x_hat_norm, T_fwd, converge_mask

    def generate_point_samples_opt(self, sdf_network, cam_locs, ray_directions, valid_mask, dists, loc, sc_factor, smpl_verts, skinning_weights, skinning_model, vol_feat, bone_transforms, trans, coord_min, coord_max, center, use_opt=False, return_sdf=True, eval_mode=False):
        ''' Backward mapping interface, will call either NN-based backward skinning or root-finding-based backward skinning '''
        n_pts = dists.size(1)
        if n_pts == 0:
            raise ValueError('No valid depth.')

        batch_size = dists.size(0)
        device = dists.device

        # Get 3D points from camera locations, camera rays and depth
        if len(dists.shape) == 2:
            # Normal case: dists is of shape (batch_size, n_pts)
            dists_pad = dists.unsqueeze(-1) # (batch_size, n_pts, 1)
            points = ray_directions * dists_pad + cam_locs # (batch_size, n_pts, 3)
        elif len(dists.shape) == 3:
            n_samples = dists.size(-1)
            # Special case: dists is of shape (batch_size, n_pts, n_samples), n_samples is number of depth samples per-ray
            cam_locs = cam_locs.view(batch_size, n_pts, 1, 3) # (batch_size, 1, 1, 3)
            dists_pad = dists.unsqueeze(-1) # (batch_size, n_pts, n_samples, 1)
            ray_directions_pad = ray_directions.unsqueeze(2) # (batch_size, n_pts, 1, 3)
            points = ray_directions_pad * dists_pad + cam_locs # (batch_size, n_pts, n_samples, 3)
        else:
            raise ValueError('dists is of wrong shape!')

        valid_points = points[valid_mask].unsqueeze(0)  # 1 x (n_valid_pts) x 3

        if use_opt:
            # Transform points from transformed space to canonical space using root-finding
            valid_points, valid_transforms_fwd, valid_converge_mask = \
                    self.inv_transform_points_opt(valid_points,
                                                  sdf_network,
                                                  loc,
                                                  sc_factor,
                                                  smpl_verts,
                                                  skinning_weights,
                                                  skinning_model,
                                                  vol_feat,
                                                  bone_transforms,
                                                  trans,
                                                  coord_min,
                                                  coord_max,
                                                  center,
                                                  eval_mode=eval_mode)

        else:
            # Transform points from transformed space to canonical space using nearest neighbor skinning
            valid_points, valid_transforms_fwd = \
                    self.inv_transform_points_smpl_verts(valid_points,
                                                         smpl_verts,
                                                         skinning_weights,
                                                         bone_transforms,
                                                         trans,
                                                         coord_min,
                                                         coord_max,
                                                         center)

            # nearest neighbor skinning will always converge
            valid_converge_mask = torch.ones(1, valid_points.size(1), device=device, dtype=torch.bool)

        # Scatter data back to its original shape
        if len(dists.shape) == 2:
            if return_sdf:
                valid_sdf = eval_sdf(valid_points, sdf_network, eval_mode=True)
                sdf = 1e11 * torch.ones(batch_size, n_pts, dtype=torch.float32, device=device)
                sdf.masked_scatter_(valid_mask, valid_sdf)
                sdf = sdf / 2.0 * 1.1 * (coord_max.squeeze(-1) - coord_min.squeeze(-1))

            points = torch.zeros_like(points)
            T_fwd = torch.zeros(batch_size, n_pts, 4, 4, dtype=torch.float32, device=device)
            converge_mask = torch.zeros_like(valid_mask)

            points.masked_scatter_(valid_mask.view(batch_size, n_pts, 1), valid_points)
            T_fwd.masked_scatter_(valid_mask.view(batch_size, n_pts, 1, 1), valid_transforms_fwd)
            converge_mask.masked_scatter_(valid_mask.view(batch_size, n_pts), valid_converge_mask)

            if return_sdf:
                return points, sdf, T_fwd, converge_mask
            else:
                return points, T_fwd, converge_mask
        else:
            if return_sdf:
                valid_sdf = eval_sdf(valid_points, sdf_network, eval_mode=True)
                sdf = 1e11 * torch.ones(batch_size, n_pts, n_samples, dtype=torch.float32, device=device)
                sdf.masked_scatter_(valid_mask, valid_sdf)
                sdf = sdf / 2.0 * 1.1 * (coord_max - coord_min)

            points = torch.zeros_like(points)
            T_fwd = torch.zeros(batch_size, n_pts, n_samples, 4, 4, dtype=torch.float32, device=device)
            converge_mask = torch.zeros(batch_size, n_pts, n_samples, device=device, dtype=torch.bool)

            points.masked_scatter_(valid_mask.view(batch_size, n_pts, n_samples, 1), valid_points)
            T_fwd.masked_scatter_(valid_mask.view(batch_size, n_pts, n_samples, 1, 1), valid_transforms_fwd)
            converge_mask.masked_scatter_(valid_mask, valid_converge_mask)

            if return_sdf:
                return points, sdf, T_fwd, converge_mask
            else:
                return points, T_fwd, converge_mask
