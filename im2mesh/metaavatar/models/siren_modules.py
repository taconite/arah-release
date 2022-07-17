'''From the SIREN repository https://github.com/vsitzmann/siren
'''

import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math

from im2mesh import hyperlayers

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            if in_features == 3:
                self.positional_encoding_coords = PosEncodingNeRF(in_features=in_features,
                                                                  sidelength=kwargs.get('sidelength', None),
                                                                  fn_samples=kwargs.get('fn_samples', None),
                                                                  use_nyquist=kwargs.get('use_nyquist', True))

                in_features = self.positional_encoding_coords.out_dim
            elif in_features == 95:
                self.positional_encoding_coords = PosEncodingNeRF(in_features=3,
                                                                  sidelength=kwargs.get('sidelength', None),
                                                                  fn_samples=kwargs.get('fn_samples', None),
                                                                  use_nyquist=kwargs.get('use_nyquist', True))

                in_features = self.positional_encoding_coords.out_dim + 92
            else:
                raise ValueError('Input feature size {} not supported!'.format(in_features))

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        # print(self)

    def forward(self, model_input, testing=False, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        if not testing:
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
        else:
            coords_org = model_input['coords']
            coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding_coords(coords)

        if 'cond' in model_input.keys():
            # cond = model_input['cond'].clone().detach().requires_grad_(True)
            cond_org = model_input['cond']
            cond = cond_org

            inps = torch.cat([coords, cond], dim=-1)
        else:
            inps = coords

        output = self.net(inps, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(self, num_joints=24, rel_joints=False, **kwargs):
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        self.ktree_parents =np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

        self.layer_0 = nn.Linear(9*num_joints + 3*num_joints, 6)

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(19, 19), nn.ReLU(), nn.Linear(19, 6))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, rots, Jtrs):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        return out


class HyperBVPNet(nn.Module):
    '''A hypernetwork representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2, hyper_in_ch=92,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, hierarchical_pose=False,
                 rel_joints=False, use_FiLM=False, **kwargs):
        super().__init__()
        self.mode = mode

        if type != 'sine':
            raise NotImplementedError('HyperBVPNet only supports sine activations for now')

        model_device = kwargs.get('model_device', None)
        self.use_FiLM = use_FiLM
        if use_FiLM:
            self.net = hyperlayers.HyperFCFiLM(hyper_in_ch=hyper_in_ch, hyper_num_hidden_layers=1,
                                               hyper_hidden_ch=256, in_ch=in_features,
                                               out_ch=out_features, num_hidden_layers=num_hidden_layers,
                                               hidden_ch=hidden_features, outermost_linear=True,
                                               model_device=model_device)
        else:
            self.net = hyperlayers.HyperFC(hyper_in_ch=hyper_in_ch, hyper_num_hidden_layers=1,
                                           hyper_hidden_ch=256, in_ch=in_features,
                                           out_ch=out_features, num_hidden_layers=num_hidden_layers,
                                           hidden_ch=hidden_features, outermost_linear=True,
                                           model_device=model_device)

        self.hierarchical_pose = hierarchical_pose
        if hierarchical_pose:
            self.pose_encoder = HierarchicalPoseEncoder(rel_joints=rel_joints)

        # print(self)

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        batch_size = coords.size(0)
        n_pts = coords.size(1)

        if self.hierarchical_pose:
            if 'rots_noise' in model_input.keys():
                model_input['rots'] = model_input['rots'] + model_input['rots_noise']

            cond = self.pose_encoder(model_input['rots'], model_input['Jtrs'])
            if 'latent' in model_input.keys():
                # print ('Use geo latent code for SDF')
                latent_code = model_input['latent']
            else:
                latent_code = None
        else:
            cond_org = model_input['cond']
            cond = cond_org

        if latent_code is None:
            decoder = self.net(cond)
        elif self.use_FiLM:
            decoder = self.net(cond, latent_code)
        else:
            decoder = self.net(cond + latent_code)

        output = decoder(coords)

        params = []
        for i in range(len(decoder) - 1):
            params.append(decoder[i][0].weights.view(batch_size, -1))

        params.append(decoder[-1].weights.view(batch_size, -1))

        return {'model_in': coords_org, 'model_out': output, 'params': params, 'decoder': decoder}


class ImageDownsampling(nn.Module):
    '''Generate samples in u,v plane according to downsampling blur kernel'''

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 8
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        elif self.in_features == 92:
            # This is the case for SMPL pose vector
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def sine_init_color(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 25, np.sqrt(6 / num_input) / 25)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
