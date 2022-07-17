import torch.nn as nn

class SkinningModel(nn.Module):
    ''' Skinning model class.
    '''

    def __init__(
            self,
            skinning_decoder_fwd=None,
            **kwargs):
        ''' Initialize skinning model instance.

        Args:
            skinning_decoder_fwd (torch.nn.Module): skinning network decoder
        '''
        super().__init__()

        self.skinning_decoder_fwd = skinning_decoder_fwd

    def forward(self):
        raise NotImplementedError('You should not call the forward function of the skinning model.')

    def decode_w(self, p, c=None, forward=True, **kwargs):
        ''' Returns skinning weights for the points p.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        if forward:
            pts_W = self.skinning_decoder_fwd(p, c=c, **kwargs)
        else:
            raise ValueError('This skinning model does not have backward networks.')

        return pts_W
