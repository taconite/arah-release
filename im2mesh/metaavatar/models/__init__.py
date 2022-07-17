from im2mesh.metaavatar.models import (decoder, siren_modules)
# Decoder dictionary
decoder_dict = {
    'single_bvp': siren_modules.SingleBVPNet,
    'hyper_bvp': siren_modules.HyperBVPNet,
    'geo_mlp': decoder.SDFNetwork,
    'deformer_mlp': decoder.Deformer,
}
