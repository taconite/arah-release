'''From the SIREN repository https://github.com/vsitzmann/siren
'''

'''Pytorch implementations of hyper-network modules.'''
import torch
import math
import torch.nn as nn
import numpy as np
from pytorch_prototyping import pytorch_prototyping
import functools

from im2mesh.metaavatar.models import siren_modules

def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class LookupLayer(nn.Module):
    def __init__(self, in_ch, out_ch, num_objects):
        super().__init__()

        self.out_ch = out_ch
        self.lookup_lin = LookupLinear(in_ch,
                                       out_ch,
                                       num_objects=num_objects)
        self.norm_nl = nn.Sequential(
            nn.LayerNorm([self.out_ch], elementwise_affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, obj_idx):
        net = nn.Sequential(
            self.lookup_lin(obj_idx),
            self.norm_nl
        )
        return net

class LookupFC(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 num_objects,
                 in_ch,
                 out_ch,
                 outermost_linear=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LookupLayer(in_ch=in_ch, out_ch=hidden_ch, num_objects=num_objects))

        for i in range(num_hidden_layers):
            self.layers.append(LookupLayer(in_ch=hidden_ch, out_ch=hidden_ch, num_objects=num_objects))

        if outermost_linear:
            self.layers.append(LookupLinear(in_ch=hidden_ch, out_ch=out_ch, num_objects=num_objects))
        else:
            self.layers.append(LookupLayer(in_ch=hidden_ch, out_ch=out_ch, num_objects=num_objects))

    def forward(self, obj_idx):
        net = []
        for i in range(len(self.layers)):
            net.append(self.layers[i](obj_idx))

        return nn.Sequential(*net)


class LookupLinear(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 num_objects):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.hypo_params = nn.Embedding(num_objects, in_ch * out_ch + out_ch)

        for i in range(num_objects):
            nn.init.kaiming_normal_(self.hypo_params.weight.data[i, :self.in_ch * self.out_ch].view(self.out_ch, self.in_ch),
                                    a=0.0,
                                    nonlinearity='relu',
                                    mode='fan_in')
            self.hypo_params.weight.data[i, self.in_ch * self.out_ch:].fill_(0.)

    def forward(self, obj_idx):
        hypo_params = self.hypo_params(obj_idx)

        # Indices explicit to catch erros in shape of output layer
        weights = hypo_params[..., :self.in_ch * self.out_ch]
        biases = hypo_params[..., self.in_ch * self.out_ch:(self.in_ch * self.out_ch)+self.out_ch]

        biases = biases.view(*(biases.size()[:-1]), 1, self.out_ch)
        weights = weights.view(*(weights.size()[:-1]), self.out_ch, self.in_ch)

        return BatchLinear(weights=weights, biases=biases)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, pretrained_siren=False):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        if pretrained_siren:
            with torch.no_grad():
                self.network[-1].weight.fill_(0.0)
                self.network[-1].bias[:map_output_dim // 2] = 1.0
                self.network[-1].bias[map_output_dim // 2:] = 0.0
        else:
            with torch.no_grad():
                self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        # print (frequencies, phase_shifts)

        return frequencies, phase_shifts


class HyperLayer(nn.Module):
    '''A hypernetwork that predicts a single Dense Layer, including sine nonlinearity.'''
    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hypo_params_init=None):
        super().__init__()

        self.hyper_linear = HyperLinear(in_ch=in_ch,
                                        out_ch=out_ch,
                                        hyper_in_ch=hyper_in_ch,
                                        hyper_num_hidden_layers=hyper_num_hidden_layers,
                                        hyper_hidden_ch=hyper_hidden_ch,
                                        hypo_params_init=hypo_params_init)
        self.nl = nn.Sequential(
            siren_modules.Sine()
        )

    def forward(self, hyper_input):
        '''
        :param hyper_input: input to hypernetwork.
        :return: nn.Module; predicted fully connected network.
        '''
        return nn.Sequential(self.hyper_linear(hyper_input), self.nl)


class HyperLayerFiLM(nn.Module):
    '''A hypernetwork that predicts a single Dense Layer, including sine nonlinearity.'''
    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hypo_params_init=None):
        super().__init__()

        self.hyper_linear = HyperLinearFiLM(in_ch=in_ch,
                                            out_ch=out_ch,
                                            hyper_in_ch=hyper_in_ch,
                                            hyper_num_hidden_layers=hyper_num_hidden_layers,
                                            hyper_hidden_ch=hyper_hidden_ch,
                                            hypo_params_init=hypo_params_init)
        self.nl = nn.Sequential(
            siren_modules.Sine()
        )

    def forward(self, hyper_input, freq, phase_shift):
        '''
        :param hyper_input: input to hypernetwork.
        :return: nn.Module; predicted fully connected network.
        '''
        return nn.Sequential(self.hyper_linear(hyper_input, freq, phase_shift), self.nl)


class HyperFCFiLM(nn.Module):
    '''Builds a hypernetwork that predicts a fully connected neural network.
    '''
    def __init__(self,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hidden_ch,
                 num_hidden_layers,
                 in_ch,
                 out_ch,
                 outermost_linear=False,
                 initial_model=None,
                 model_device=None):
        super().__init__()

        PreconfHyperLinear = partialclass(HyperLinear,
                                          hyper_in_ch=hyper_in_ch,
                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
                                          hyper_hidden_ch=hyper_hidden_ch)
        PreconfHyperLayer = partialclass(HyperLayerFiLM,
                                         hyper_in_ch=hyper_in_ch,
                                         hyper_num_hidden_layers=hyper_num_hidden_layers,
                                         hyper_hidden_ch=hyper_hidden_ch)

        self.hidden_ch = hidden_ch
        self.layers = nn.ModuleList()

        if initial_model is not None:
            # state_dict = torch.load('out/ptf-plus/ptf_ptfs-pointnet2_smlp-loopreg_CAPE-SV-wo-raw_1gpus/model_20000.pt')['model']
            if model_device is not None:
                state_dict = torch.load(initial_model, map_location=model_device)['model']
            else:
                state_dict = torch.load(initial_model, map_location='cpu')['model']

            hypo_params_init = torch.cat([state_dict['decoder.net.net.0.0.weight'].view(-1), state_dict['decoder.net.net.0.0.bias'].view(-1)], dim=0).view(1, -1)
            hypo_params_init = hypo_params_init.clone().detach().requires_grad_(False)

            self.layers.append(PreconfHyperLayer(in_ch=in_ch, out_ch=hidden_ch, hypo_params_init=hypo_params_init))

            for i in range(num_hidden_layers):
                hypo_params_init = torch.cat([state_dict['decoder.net.net.{}.0.weight'.format(i+1)].view(-1),
                                              state_dict['decoder.net.net.{}.0.bias'.format(i+1)].view(-1)],
                                             dim=0).view(1, -1)
                hypo_params_init = hypo_params_init.clone().detach().requires_grad_(False)
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=hidden_ch, hypo_params_init=hypo_params_init))

            hypo_params_init = torch.cat([state_dict['decoder.net.net.{}.0.weight'.format(num_hidden_layers+1)].view(-1),
                                          state_dict['decoder.net.net.{}.0.bias'.format(num_hidden_layers+1)].view(-1)],
                                         dim=0).view(1, -1)
            hypo_params_init = hypo_params_init.clone().detach().requires_grad_(False)

            if outermost_linear:
                self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=out_ch, hypo_params_init=hypo_params_init))
            else:
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=out_ch, hypo_params_init=hypo_params_init))

        else:
            self.layers.append(PreconfHyperLayer(in_ch=in_ch, out_ch=hidden_ch))

            for i in range(num_hidden_layers):
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=hidden_ch))

            if outermost_linear:
                self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=out_ch))
            else:
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=out_ch))

        self.mapping_network = CustomMappingNetwork(128, 256, (len(self.layers) - 1) * hidden_ch * 2, True)

    def forward(self, hyper_input, latent_code):
        '''
        :param hyper_input: Input to hypernetwork.
        :return: nn.Module; Predicted fully connected neural network.
        '''
        net = []
        freqs, phase_shifts = self.mapping_network(latent_code)
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                start = i * self.hidden_ch
                end = (i+1) * self.hidden_ch
                net.append(self.layers[i](hyper_input, freqs[..., start:end], phase_shifts[..., start:end]))
            else:
                net.append(self.layers[i](hyper_input))

        return nn.Sequential(*net)


class HyperFC(nn.Module):
    '''Builds a hypernetwork that predicts a fully connected neural network.
    '''
    def __init__(self,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hidden_ch,
                 num_hidden_layers,
                 in_ch,
                 out_ch,
                 outermost_linear=False,
                 initial_model=None,
                 model_device=None):
        super().__init__()

        PreconfHyperLinear = partialclass(HyperLinear,
                                          hyper_in_ch=hyper_in_ch,
                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
                                          hyper_hidden_ch=hyper_hidden_ch)
        PreconfHyperLayer = partialclass(HyperLayer,
                                          hyper_in_ch=hyper_in_ch,
                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
                                          hyper_hidden_ch=hyper_hidden_ch)

        self.layers = nn.ModuleList()

        if initial_model is not None:
            # state_dict = torch.load('out/ptf-plus/ptf_ptfs-pointnet2_smlp-loopreg_CAPE-SV-wo-raw_1gpus/model_20000.pt')['model']
            if model_device is not None:
                state_dict = torch.load(initial_model, map_location=model_device)['model']
            else:
                state_dict = torch.load(initial_model)['model']

            hypo_params_init = torch.cat([state_dict['decoder.net.net.0.0.weight'].view(-1), state_dict['decoder.net.net.0.0.bias'].view(-1)], dim=0).view(1, -1)
            hypo_params_init = hypo_params_init.clone().detach().requires_grad_(False)

            self.layers.append(PreconfHyperLayer(in_ch=in_ch, out_ch=hidden_ch, hypo_params_init=hypo_params_init))

            for i in range(num_hidden_layers):
                hypo_params_init = torch.cat([state_dict['decoder.net.net.{}.0.weight'.format(i+1)].view(-1),
                                              state_dict['decoder.net.net.{}.0.bias'.format(i+1)].view(-1)],
                                             dim=0).view(1, -1)
                hypo_params_init = hypo_params_init.clone().detach().requires_grad_(False)
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=hidden_ch, hypo_params_init=hypo_params_init))

            hypo_params_init = torch.cat([state_dict['decoder.net.net.{}.0.weight'.format(num_hidden_layers+1)].view(-1),
                                          state_dict['decoder.net.net.{}.0.bias'.format(num_hidden_layers+1)].view(-1)],
                                         dim=0).view(1, -1)
            hypo_params_init = hypo_params_init.clone().detach().requires_grad_(False)

            if outermost_linear:
                self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=out_ch, hypo_params_init=hypo_params_init))
            else:
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=out_ch, hypo_params_init=hypo_params_init))

        else:
            self.layers.append(PreconfHyperLayer(in_ch=in_ch, out_ch=hidden_ch))

            for i in range(num_hidden_layers):
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=hidden_ch))

            if outermost_linear:
                self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=out_ch))
            else:
                self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=out_ch))


    def forward(self, hyper_input):
        '''
        :param hyper_input: Input to hypernetwork.
        :return: nn.Module; Predicted fully connected neural network.
        '''
        net = []
        for i in range(len(self.layers)):
            net.append(self.layers[i](hyper_input))

        return nn.Sequential(*net)


class BatchLinear(nn.Module):
    def __init__(self,
                 weights,
                 biases):
        '''Implements a batch linear layer.

        :param weights: Shape: (batch, out_ch, in_ch)
        :param biases: Shape: (batch, 1, out_ch)
        '''
        super().__init__()

        self.weights = weights
        self.biases = biases

    def __repr__(self):
        return "BatchLinear(in_ch=%d, out_ch=%d)"%(self.weights.shape[-1], self.weights.shape[-2])

    def forward(self, input):
        output = input.matmul(self.weights.permute(*[i for i in range(len(self.weights.shape)-2)], -1, -2))
        output += self.biases
        return output


class BatchLinearFiLM(nn.Module):
    def __init__(self,
                 weights,
                 biases,
                 freq,
                 phase_shift):
        '''Implements a batch linear layer with FiLM.

        :param weights: Shape: (batch, out_ch, in_ch)
        :param biases: Shape: (batch, 1, out_ch)
        '''
        super().__init__()

        self.weights = weights
        self.biases = biases
        self.freq = freq
        self.phase_shift = phase_shift

    def __repr__(self):
        return "BatchLinear(in_ch=%d, out_ch=%d)"%(self.weights.shape[-1], self.weights.shape[-2])

    def forward(self, input):
        output = input.matmul(self.weights.permute(*[i for i in range(len(self.weights.shape)-2)], -1, -2))
        output += self.biases
        return self.freq * output + self.phase_shift


def last_hyper_layer_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        # Initialization last layer to be 0s, so that initial prediction of residual
        # is always 0
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


class HyperLinear(nn.Module):
    '''A hypernetwork that predicts a single linear layer (weights & biases).'''
    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hypo_params_init=None):

        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if hypo_params_init is None:
            hypo_params_init = torch.zeros(1, (in_ch * out_ch) + out_ch, dtype=torch.float32)

        self.register_buffer('hypo_params_init', hypo_params_init)

        self.hypo_params = pytorch_prototyping.FCBlock(in_features=hyper_in_ch,
                                                       hidden_ch=hyper_hidden_ch,
                                                       num_hidden_layers=hyper_num_hidden_layers,
                                                       out_features=(in_ch * out_ch) + out_ch,
                                                       outermost_linear=True)

        self.hypo_params[-1].apply(last_hyper_layer_init)

    def forward(self, hyper_input):
        hypo_params = self.hypo_params(hyper_input)

        if self.hypo_params_init is not None:
            hypo_params += self.hypo_params_init

        # Indices explicit to catch erros in shape of output layer
        weights = hypo_params[..., :self.in_ch * self.out_ch]
        biases = hypo_params[..., self.in_ch * self.out_ch:(self.in_ch * self.out_ch)+self.out_ch]

        biases = biases.view(*(biases.size()[:-1]), 1, self.out_ch)
        weights = weights.view(*(weights.size()[:-1]), self.out_ch, self.in_ch)

        return BatchLinear(weights=weights, biases=biases)


class HyperLinearFiLM(nn.Module):
    '''A hypernetwork that predicts a single linear layer (weights & biases).'''
    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hypo_params_init=None):

        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if hypo_params_init is None:
            hypo_params_init = torch.zeros(1, (in_ch * out_ch) + out_ch, dtype=torch.float32)

        self.register_buffer('hypo_params_init', hypo_params_init)

        self.hypo_params = pytorch_prototyping.FCBlock(in_features=hyper_in_ch,
                                                       hidden_ch=hyper_hidden_ch,
                                                       num_hidden_layers=hyper_num_hidden_layers,
                                                       out_features=(in_ch * out_ch) + out_ch,
                                                       outermost_linear=True)


        self.hypo_params[-1].apply(last_hyper_layer_init)

    def forward(self, hyper_input, freq, phase_shift):
        hypo_params = self.hypo_params(hyper_input)

        if self.hypo_params_init is not None:
            hypo_params += self.hypo_params_init

        # Indices explicit to catch erros in shape of output layer
        weights = hypo_params[..., :self.in_ch * self.out_ch]
        biases = hypo_params[..., self.in_ch * self.out_ch:(self.in_ch * self.out_ch)+self.out_ch]

        biases = biases.view(*(biases.size()[:-1]), 1, self.out_ch)
        weights = weights.view(*(weights.size()[:-1]), self.out_ch, self.in_ch)

        return BatchLinearFiLM(weights=weights, biases=biases, freq=freq, phase_shift=phase_shift)
