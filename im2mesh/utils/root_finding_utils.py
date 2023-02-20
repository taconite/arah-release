import os, sys
import torch
import pytorch3d
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import im2mesh.utils.diff_operators as diff_operators
from im2mesh.utils.broyden import broyden
from im2mesh.utils.utils import hierarchical_softmax
from collections import OrderedDict

''' Copied from SNARF '''
def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    # p:n_point, n:n_bone, i,k: n_dim+1
    w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
    if inverse:
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        # x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf, x_h)

    return (x_h[:, :, :3], w_tf)
''' End of copying '''


def normalize_canonical_points(pts, coord_min, coord_max, center):
    pts = pts - center
    padding = (coord_max - coord_min) * 0.05
    pts = (pts - coord_min + padding) / (coord_max - coord_min) / 1.1
    pts = pts - 0.5
    pts = pts * 2.

    return pts


def unnormalize_canonical_points(pts, coord_min, coord_max, center):
    padding = (coord_max - coord_min) * 0.05
    pts = (pts / 2.0 + 0.5) * 1.1 * (coord_max - coord_min) + coord_min - padding +  center

    return pts


def query_weights(x_hat, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, mask=None, point_batch_size=100000):
    """Canonical point -> deformed point

    Args:
        x_hat (B x N x 3 float Tensor): points in canonical space
        loc (B x 1 x 3 float Tensor): center for normalizing x_hat
        sc_factor (B x 1 x 1 float Tensor): scaling factor for normalizing x_hat
        skinning_model (Torch.nn.Module): skinning network
        vol_feat (float Tensor): conditional volumetric features for skinning network
        mask ((B*N) bool Tensor): valid mask

    Returns:
        w (B x N x 24 float Tensor): skinning weights for each point
    """

    batch_size, n_pts, _ = x_hat.size()

    if mask is not None:
        mask = mask.view(batch_size, n_pts)

    if isinstance(vol_feat, OrderedDict):
        x_hat_norm = (x_hat - loc) * sc_factor
    else:
        x_hat_norm = normalize_canonical_points(x_hat, coord_min, coord_max, center)

    if mask is not None:
        # TODO: to use mask here,  we need to assume the batch is either formed of multi-view
        # images of the same frame, or the skinning network does not condition on any latent code
        x_hat_norm = x_hat_norm[mask].unsqueeze(0)    # only query on valid points to save computation

    if isinstance(vol_feat, OrderedDict):
        vol_feat_ = OrderedDict()
        for k, v in vol_feat.items():
            vol_feat_[k] = v[:1, ...]

        vol_feat = vol_feat_

    p_split = torch.split(x_hat_norm.reshape(1, -1, 3), point_batch_size, dim=1)
    w = []
    for pi in p_split:
        wi = skinning_model.decode_w(pi, c=torch.empty(pi.size(0), 0, device=pi.device, dtype=torch.float32), forward=True)

        if wi.size(-1) == 24:
            w.append(F.softmax(wi, dim=-1)) # naive softmax
        elif wi.size(-1) == 25:
            w.append(hierarchical_softmax(wi * 20)) # hierarchical softmax in SNARF
        else:
            raise ValueError('Wrong output size of skinning network. Expected 24 or 25, got {}.'.format(wi.size(-1)))

    w = torch.cat(w, dim=1)

    if mask is not None:
        # w = F.softmax(w, dim=-1)
        w_ret = torch.zeros(batch_size, n_pts, 24, device=x_hat.device, dtype=x_hat.dtype)
        w_ret.masked_scatter_(mask.unsqueeze(-1), w)
    else:
        w_ret = w.reshape(batch_size, n_pts, -1)
        # w_ret = F.softmax(w, dim=-1)

    return w_ret


def eval_sdf(points, sdf_model, eval_mode=False, point_batch_size=100000):
    if len(points.shape) == 3:
        batch_size, n_pts, _ = points.size()
    else:
        n_pts, _ = points.size()
        batch_size = 1

    p_split = torch.split(points.reshape(1, -1, 3), point_batch_size, dim=1)
    sdf_vals = []
    for pi in p_split:
        if eval_mode:
            # There is some weird bug from either PyTorch and PyTorch-Lightning,
            # in which memory usage still grows very fast even with gradient disabled
            # when evaluating MLPs with batches of points.
            # Detaching the computation result and convert it to CPU tensors somehow
            # prevents this weird bug
            with torch.no_grad():
                pi = pi.detach()
                sdf_val = sdf_model(pi).detach().cpu()
                sdf_vals.append(sdf_val)
                del sdf_val
        else:
            sdf_val = sdf_model(pi)
            sdf_vals.append(sdf_val)

    if eval_mode:
        return torch.cat(sdf_vals, dim=1).reshape(batch_size, n_pts, 1).to(points.device)
    else:
        return torch.cat(sdf_vals, dim=1).reshape(batch_size, n_pts, 1)


def forward_skinning(x_hat, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, bone_transforms, mask=None, return_w=False, point_batch_size=100000):
    """Canonical point -> deformed point

    Args:
        x_hat (B x N x 3 float Tensor): points in canonical space
        loc (B x 1 x 3 float Tensor): center for normalizing x_hat
        sc_factor (B x 1 x 1 float Tensor): scaling factor for normalizing x_hat
        skinning_model (Torch.nn.Module): skinning network
        vol_feat (float Tensor): conditional volumetric features for skinning network
        bone_transforms (B x 24 x 4 x 4 float Tensor): SMPL bone transforms
        mask ((B*N) bool Tensor): valid mask

    Returns:
        x_bar (B x N x 3 float Tensor): points in transformed space
    """
    w = query_weights(x_hat, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, mask=mask, point_batch_size=point_batch_size)
    x_bar, T = skinning(x_hat, w, bone_transforms, inverse=False)
    if return_w:
        return x_bar, T, w
    else:
        return x_bar, T


def forward_skinning_jac(x_hat, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, bone_transforms, mask=None, point_batch_size=100000):
    """Compute Jacobian of forward transformation

    Args:
        x_hat (B x N x 3 float Tensor): points in canonical space
        loc (B x 1 x 3 float Tensor): center for normalizing x_hat
        sc_factor (B x 1 x 1 float Tensor): scaling factor for normalizing x_hat
        skinning_model (Torch.nn.Module): skinning network
        vol_feat (float Tensor): conditional volumetric features for skinning network
        bone_transforms (B x 24 x 4 x 4 float Tensor): SMPL bone transforms
        mask ((B*N) bool Tensor): valid mask

    Returns:
        jac (B x N x 3 x 3 float Tensor): Jacobian of forward skinning function
    """
    batch_size = x_hat.size(0)
    if mask is None:
        p_split = torch.split(x_hat, point_batch_size, dim=1)
        jacs = []
        for pi_hat in p_split:
            with torch.enable_grad():
                pi_hat.requires_grad_(True)

                pi_bar, _ = forward_skinning(pi_hat, loc, sc_factor,
                                             coord_min, coord_max, center,
                                             skinning_model, vol_feat,
                                             bone_transforms, point_batch_size=point_batch_size)

                jac, status = diff_operators.jacobian(pi_bar,
                                                      pi_hat)

                pi_hat.requires_grad_(False)
                jacs.append(jac)

        jacs = torch.cat(jacs, dim=1) #.reshape(batch_size, -1, 3, 3)
    else:
        p_split = torch.split(x_hat, point_batch_size, dim=1)
        m_split = torch.split(mask, point_batch_size, dim=1)
        jacs = []
        for pi_hat, mi in zip(p_split, m_split):
            with torch.enable_grad():
                pi_hat.requires_grad_(True)

                pi_bar, _ = forward_skinning(pi_hat, loc, sc_factor,
                                             coord_min, coord_max, center,
                                             skinning_model, vol_feat,
                                             bone_transforms, mask=mi, point_batch_size=point_batch_size)

                jac, status = diff_operators.jacobian(pi_bar,
                                                      pi_hat)

                pi_hat.requires_grad_(False)
                jacs.append(jac)

        jacs = torch.cat(jacs, dim=1) #.reshape(batch_size, -1, 3, 3)

    return jacs


def init_x_hat(x_bar, init_bones, bone_transforms):
    """Transform x_bar to canonical space for initialization

    Args:
        x_bar (B x N x 3 float Tensor): points in transformed space
        init_bones (B x N x K long Tensor): nearest K SMPL bones for each point in x_bar
        bone_transforms (B x 24 x 4 x 4 float Tensor): SMPL bone transforms

    Returns:
        x_hat_0 (B x N x 3 float Tensor): initial canonical correspondences for x_bar
    """
    batch_size, n_pts, _ = x_bar.size()
    _, n_joint, _, _ = bone_transforms.size()
    device = x_bar.device

    x_hat_0, w_tf = [], []
    if isinstance(init_bones, list):
        for i in range(len(init_bones)):
            w = torch.zeros((batch_size, n_pts, n_joint), device=device)
            w[..., init_bones[i]] = 1
            x_hat_0_, w_tf_ = skinning(x_bar, w, bone_transforms, inverse=True)
            x_hat_0.append(x_hat_0_)
            w_tf.append(w_tf_)
    else:
        for i in range(init_bones.size(-1)):
            w = torch.zeros((batch_size, n_pts, n_joint), device=device)
            bv, pv = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(n_pts).to(device)], indexing='ij')
            w[bv, pv, init_bones[..., i]] = 1
            x_hat_0_, w_tf_ = skinning(x_bar, w, bone_transforms, inverse=True)
            x_hat_0.append(x_hat_0_)
            w_tf.append(w_tf_)

    x_hat_0 = torch.stack(x_hat_0, dim=2)
    w_tf = torch.stack(w_tf, dim=2)

    return x_hat_0, w_tf


def search_canonical_corr(x_bar, x_hat_0, T_fwd_0, loc, sc_factor, skinning_model,
                          vol_feat, bone_transforms, coord_min, coord_max,
                          center, eval_mode, eval_point_batch_size=100000):
    """Search canonical correspondences of x_bar via root-finding
    Args:
    Returns:
    """
    # reshape to B x (N*I) x 3 for other functions
    batch_size, n_pts, n_init, n_dim = x_hat_0.size()
    if not eval_mode or eval_point_batch_size <= 0:
        x_hat_0 = x_hat_0.reshape(batch_size, n_pts * n_init, n_dim)
        x_bar_tgt = x_bar.repeat_interleave(n_init, dim=1)

        # compute init jacobians
        w = query_weights(x_hat_0, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, mask=None)
        J_inv_init = torch.einsum("bpn,bnij->bpij", w, bone_transforms)[:, :, :3, :3].inverse()
        # J_inv_init = forward_skinning_jac(x_hat_0, loc, sc_factor, skinning_model, vol_feat, bone_transforms, mask=None)

        # reshape init to [?,D,...] for boryden
        x_hat_0 = x_hat_0.reshape(-1, n_dim, 1)
        T_fwd_0 = T_fwd_0.reshape(-1, n_dim+1, n_dim+1)
        J_inv_init = J_inv_init.flatten(0, 1)

        # construct function for root finding
        def _func(x_hat_opt, mask=None):
            # reshape to [B,?,D] for other functions
            x_hat_opt = x_hat_opt.reshape(batch_size, n_pts * n_init, n_dim)
            x_bar_opt, T_opt = forward_skinning(x_hat_opt, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, bone_transforms, mask=mask)
            error = x_bar_opt - x_bar_tgt
            # reshape to [?,D,1] for boryden
            error = error.flatten(0, 1)[mask].unsqueeze(-1)
            T_opt = T_opt.flatten(0, 1)[mask]
            return error, T_opt

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, x_hat_0, T_fwd_0, J_inv_init)

        # reshape back to [B,N,I,D]
        x_hat_opt = result["result"].reshape(batch_size, n_pts, n_init, n_dim)
        T_fwd_opt = result["transforms"].reshape(batch_size, n_pts, n_init, 4, 4)
        valid_mask = result["valid_ids"].reshape(batch_size, n_pts, n_init)
        error_opt = result["diff"].reshape(batch_size, n_pts, n_init)
    else:
        assert (batch_size == 1)
        x_hat_0 = x_hat_0.reshape(batch_size, n_pts * n_init, n_dim)
        T_fwd_0 = T_fwd_0.reshape(batch_size, n_pts * n_init, n_dim+1, n_dim+1)
        x_bar_tgt = x_bar.repeat_interleave(n_init, dim=1)

        x_hat_0_split = torch.split(x_hat_0, eval_point_batch_size, dim=1)
        x_bar_tgt_split = torch.split(x_bar_tgt, eval_point_batch_size, dim=1)
        T_fwd_0_split = torch.split(T_fwd_0, eval_point_batch_size, dim=1)

        x_hat_opts = []
        T_fwd_opts = []
        valid_masks = []
        error_opts = []

        for x_hat_0, x_bar_tgt, T_fwd_0 in zip(x_hat_0_split, x_bar_tgt_split, T_fwd_0_split):
            # compute init jacobians
            w = query_weights(x_hat_0, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, mask=None, point_batch_size=eval_point_batch_size)
            J_inv_init = torch.einsum("bpn,bnij->bpij", w, bone_transforms)[:, :, :3, :3].inverse()
            # J_inv_init = forward_skinning_jac(x_hat_0, loc, sc_factor, skinning_model, vol_feat, bone_transforms, mask=None)

            # reshape init to [?,D,...] for boryden
            x_hat_0 = x_hat_0.reshape(-1, n_dim, 1)
            T_fwd_0 = T_fwd_0.reshape(-1, n_dim+1, n_dim+1)
            J_inv_init = J_inv_init.flatten(0, 1)

            # construct function for root finding
            def _func(x_hat_opt, mask=None):
                # reshape to [B,?,D] for other functions
                x_hat_opt = x_hat_opt.reshape(batch_size, -1, n_dim)
                x_bar_opt, T_opt = forward_skinning(x_hat_opt, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, bone_transforms, mask=mask, point_batch_size=eval_point_batch_size)
                error = x_bar_opt - x_bar_tgt
                # reshape to [?,D,1] for boryden
                error = error.flatten(0, 1)[mask].unsqueeze(-1)
                T_opt = T_opt.flatten(0, 1)[mask]
                return error, T_opt

            # run broyden without grad
            with torch.no_grad():
                result = broyden(_func, x_hat_0, T_fwd_0, J_inv_init)

            # reshape back to [B,N,I,D]
            x_hat_opts.append(result["result"].reshape(batch_size, -1, n_dim))
            T_fwd_opts.append(result["transforms"].reshape(batch_size, -1, 4, 4))
            valid_masks.append(result["valid_ids"].reshape(batch_size, -1))
            error_opts.append(result["diff"].reshape(batch_size, -1))

        x_hat_opt = torch.cat(x_hat_opts, dim=1).reshape(batch_size, n_pts, n_init, n_dim)
        T_fwd_opt = torch.cat(T_fwd_opts, dim=1).reshape(batch_size, n_pts, n_init, 4, 4)
        valid_mask = torch.cat(valid_masks, dim=1).reshape(batch_size, n_pts, n_init)
        error_opt = torch.cat(error_opts, dim=1).reshape(batch_size, n_pts, n_init)

    return x_hat_opt, T_fwd_opt, valid_mask, error_opt


def search_iso_surface_depth(cam_pos, cam_rays, valid_mask, x_hat_0, Zdepth_0, T_fwd_0,
                             sdf_model, loc, sc_factor, skinning_model, vol_feat,
                             bone_transforms, trans, coord_min, coord_max, center, eval_mode):
    """ Search isosurface points of canonical SDF and depth in posed space via joint root-finding
    Args:
    Returns:
    """
    # reshape to B x (N*I) x 3 for other functions
    batch_size, n_pts, n_dim = x_hat_0.size()
    n_init = 1
    x_hat_0 = x_hat_0.reshape(batch_size, n_pts * n_init, n_dim)
    Zdepth_0 = Zdepth_0.reshape(batch_size, n_pts * n_init, 1)

    inputs_0 = torch.cat([x_hat_0, Zdepth_0], dim=-1)

    cam_pos_pad = cam_pos[valid_mask].unsqueeze(0) # (1, N, 3)

    if valid_mask.sum() == 0:
        x_hat_opt = x_hat_0.clone()
        Zdepth_opt = Zdepth_0.view(batch_size, n_pts).clone()
        T_fwd_opt = T_fwd_0.clone()
        converge_mask_opt = valid_mask.clone()

        return x_hat_opt, Zdepth_opt, T_fwd_opt, converge_mask_opt

    valid_indices = torch.nonzero(valid_mask)

    lengths = valid_mask.sum(-1)
    n_valid_pts = lengths.max()
    dst_indices = torch.cat([torch.arange(length.item()) for length in lengths], dim=0)

    valid_inputs_0 = inputs_0[valid_mask].unsqueeze(0)
    valid_cam_rays = cam_rays[valid_mask].unsqueeze(0)
    valid_T_fwd_0 = T_fwd_0[valid_mask].unsqueeze(0)
    valid_x_hat_0 = valid_inputs_0[..., :3]

    # compute init jacobians
    if not eval_mode:
        raise NotImplementedError('Training mode of marching-SNARF is not implemented yet')
    else:
        # Jacobian of forward skinning
        J_lbs = forward_skinning_jac(valid_x_hat_0, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, bone_transforms, mask=None)
        # Jacobian of SDF
        with torch.enable_grad():
            x_hat_0_ = valid_x_hat_0.clone().detach().requires_grad_(True)
            x_hat_0_norm = normalize_canonical_points(x_hat_0_, coord_min, coord_max, center)
            pred_sdf = eval_sdf(x_hat_0_norm, sdf_model, eval_mode=False)
            pred_sdf = pred_sdf / 2.0 * 1.1 * (coord_max - coord_min)   # convert from normalized SDF space to SMPL canonical space
            grad_sdf = diff_operators.gradient_no_diff(pred_sdf, x_hat_0_).unsqueeze(-2)

        # Now form the final Jacobian matrix for g
        J_init = torch.cat([grad_sdf, J_lbs], dim=-2)
        J_init = torch.cat([J_init, F.pad(-valid_cam_rays, (1, 0)).unsqueeze(-1)], dim=-1)
        J_inv_init = J_init.inverse()

    # reshape init to [?,D,...] for boryden
    valid_inputs_0 = valid_inputs_0.reshape(-1, n_dim+1, 1)
    valid_T_fwd_0 = valid_T_fwd_0.reshape(-1, n_dim+1, n_dim+1)
    J_inv_init = J_inv_init.flatten(0, 1)

    # construct function for root finding
    def _func(inputs, mask=None):
        # Disentangle inputs to x_hat and Zdepth
        x_hat_opt = inputs[:, :3, :]
        Zdepth_opt = inputs[:, 3:, :]
        # reshape to [B,?,D] for other functions
        x_hat_opt = x_hat_opt.reshape(batch_size, n_valid_pts * n_init, n_dim)
        Zdepth_opt = Zdepth_opt.reshape(batch_size, n_valid_pts * n_init, 1)

        # Form x_bar_tgt from camera center, camera rays and depths
        x_bar = valid_cam_rays * Zdepth_opt + cam_pos_pad # (B, N, 3)
        x_bar_tgt = (x_bar - trans).repeat_interleave(n_init, dim=1)

        # Apply forward skinning
        x_bar_opt, T_opt = forward_skinning(x_hat_opt, loc, sc_factor, coord_min, coord_max, center, skinning_model, vol_feat, bone_transforms, mask=mask)

        error_corr = x_bar_opt - x_bar_tgt  # error measure for correspondence search

        # Compute SDF
        x_hat_opt_norm = normalize_canonical_points(x_hat_opt, coord_min, coord_max, center)
        mask_sdf = mask.reshape(batch_size, -1)
        pred_sdf = eval_sdf(x_hat_opt_norm[mask_sdf], sdf_model, eval_mode=True)
        error_sdf = torch.zeros_like(Zdepth_opt)
        error_sdf.masked_scatter_(mask_sdf.unsqueeze(-1), pred_sdf)   # error measure for isosurface search
        error_sdf = error_sdf / 2.0 * 1.1 * (coord_max - coord_min)   # convert from normalized SDF space to canonical space

        # Concatanate error vectors
        error = torch.cat([error_sdf, error_corr], dim=-1)

        # reshape to [?,D,1] for boryden
        error = error.flatten(0, 1)[mask].unsqueeze(-1)
        T_opt = T_opt.flatten(0, 1)[mask]
        return error, T_opt

    # run broyden without grad
    with torch.no_grad():
        result = broyden(_func, valid_inputs_0, valid_T_fwd_0, J_inv_init)

    # reshape back to [B,N,I,D]
    outputs = result["result"].reshape(batch_size, n_valid_pts, n_dim+1)
    valid_x_hat_opt = outputs[..., :3]
    valid_Zdepth_opt = outputs[..., 3]
    valid_T_fwd_opt = result["transforms"].reshape(batch_size, n_valid_pts, 4, 4)
    valid_converge_mask = result["valid_ids"].reshape(batch_size, n_valid_pts)

    x_hat_opt = x_hat_0.clone()
    Zdepth_opt = Zdepth_0.view(batch_size, n_pts).clone()
    T_fwd_opt = T_fwd_0.clone()
    converge_mask_opt = valid_mask.clone()

    # Construct mask first
    converge_mask_opt.masked_scatter_(valid_mask, valid_converge_mask[valid_indices[:, 0], dst_indices])

    # Scatter results into result tensors
    x_hat_opt.masked_scatter_(valid_mask.view(batch_size, n_pts, 1), valid_x_hat_opt[valid_indices[:, 0], dst_indices])

    Zdepth_opt.masked_scatter_(valid_mask, valid_Zdepth_opt[valid_indices[:, 0], dst_indices])
    T_fwd_opt.masked_scatter_(valid_mask.view(batch_size, n_pts, 1, 1), valid_T_fwd_opt[valid_indices[:, 0], dst_indices])

    return x_hat_opt, Zdepth_opt, T_fwd_opt, converge_mask_opt
