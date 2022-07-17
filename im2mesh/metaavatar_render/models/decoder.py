import torch
import torch.nn as nn
import torch.nn.functional as F

from im2mesh.metaavatar_render.models.embedder import get_embedder
from im2mesh.metaavatar.models.siren_modules import HierarchicalPoseEncoder
from im2mesh.hyperlayers import CustomMappingNetwork

# This implementation is borrowed and modified from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires=0,
                 multires_view=0,
                 skips=[],
                 squeeze_out=True,
                 rel_joints=True,
                 pose_encoder='leap'):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn = None
        self.embedview_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)

        if multires_view > 0:
            embedview_fn, input_ch_view = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch_view - 3)

        self.pose_encoder_type = pose_encoder
        if pose_encoder in ['leap']:
            self.pose_encoder = HierarchicalPoseEncoder(rel_joints=rel_joints)

        self.skips = skips
        for skip in skips:
            dims[skip] = dims[skip] // 2 + dims[0]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in skips:
                out_dim = (dims[l + 1] - dims[0])
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, sdf_feature, pose_feature):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        n_pts = sdf_feature.size(0)
        if self.pose_encoder_type == 'leap':
            pose_feature = self.pose_encoder(pose_feature['rots_full'][:1], pose_feature['Jtrs_posed'][:1])
            feature_vectors = torch.cat([sdf_feature, pose_feature.repeat(n_pts, 1)], dim=-1)
        elif self.pose_encoder_type in ['root', 'hybrid']:
            if 'rot_noise' in pose_feature.keys() and 'trans_noise' in pose_feature.keys():
                rot = pose_feature['rots_full'][:1, :1].reshape(1, 9) + pose_feature['rot_noise']
                trans = pose_feature['Jtrs_posed'][:1, :1].reshape(1, 3) + pose_feature['trans_noise']
            else:
                rot = pose_feature['rots_full'][:1, :1].reshape(1, 9)
                trans = pose_feature['Jtrs_posed'][:1, :1].reshape(1, 3)

            pose_feat = torch.cat([rot, trans], dim=-1)    # 1 x 12 pose feature
            if self.pose_encoder_type == 'hybrid':
                pose_feat = torch.cat([pose_feat, pose_feature['latent_code']], dim=-1) # 1 x (12 + 128) pose feature

            feature_vectors = torch.cat([sdf_feature, pose_feat.repeat(n_pts, 1)], dim=-1)
        elif self.pose_encoder_type == 'latent':
            pose_feat = pose_feature['latent_code'] # 1 x 128 pose feature

            feature_vectors = torch.cat([sdf_feature, pose_feat.repeat(n_pts, 1)], dim=-1)
        else:
            feature_vectors = sdf_feature

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skips:
                # print ('dim1: {} dim2: {}'.format(rendering_input.size(-1), x.size(-1)))
                x = lin(torch.cat([rendering_input, x], dim=-1))
            else:
                x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones_like(x) * torch.linalg.norm(self.variance) # prevent variance from going negative
