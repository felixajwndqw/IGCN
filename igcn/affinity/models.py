import torch
import torch.nn as nn

from igcn.cmplx import concatenate, new_cmplx
from igcn.seg.cmplx_modules import TripleIGConvCmplx
from igcn.affinity.modules import Down3D, Up3D
from igcn.utils import _trio


def cat_activations(x, y):
    # for complex tensors [2, B, C, W, H]
    return torch.cat((x, y), 2)


COMB_FNS = {
    'cat': {
        'fn': cat_activations,
        'cat': True
    },
    'add': {
        'fn': torch.add,
        'cat': False
    }
}


class Gabor3D(nn.Module):
    """
    3D U-Net architecture with Gabor convs.
    Modifed from github.com/inferno-pytorch/neurofire
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 scale_factor=2,
                 final_activation='auto',
                 comb_fn='cat',
                 conv_type_key='vanilla',
                 pooling='avg',
                 gp='max',
                 **conv_kwargs):
        """
        Parameter:
        ----------
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        initial_num_fmaps (int): number of feature maps of the first layer
        fmap_growth (int): growth factor of the feature maps; the number of feature maps
        in layer k is given by initial_num_fmaps * fmap_growth**k
        final_activation:  final activation used
        scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
        final_activation:  final activation used (default: 'auto')
        conv_type_key: convolutin type
        """
        super().__init__()
        
        print('out_channels', out_channels)

        # validate scale factor
        assert isinstance(scale_factor, (int, list, tuple))
        self.scale_factor = [scale_factor] * 3 if isinstance(scale_factor, int) else scale_factor
        assert len(self.scale_factor) == 3
        # NOTE individual scale factors can have multiple entries for anisotropic sampling
        assert all(isinstance(sfactor, (int, list, tuple))
                   for sfactor in self.scale_factor)

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gp = gp

        # Build encoders with proper number of feature maps
        f0e = initial_num_fmaps
        f1e = initial_num_fmaps * fmap_growth
        f2e = initial_num_fmaps * fmap_growth**2
        print(f0e, f1e, f2e)
        encoders = [
            TripleIGConvCmplx(in_channels, f0e, _trio(3), first=True, data_dim='3d', gp=None, **conv_kwargs),
            Down3D(f0e, f1e, 3, self.scale_factor[0], pooling=pooling, gp=None, **conv_kwargs),
            Down3D(f1e, f2e, 3, self.scale_factor[1], pooling=pooling, gp=None, **conv_kwargs)
        ]

        # Build base
        # number of base output feature maps
        f0b = initial_num_fmaps * fmap_growth**3
        base = Down3D(f2e, f0b, 3, scale_factor=self.scale_factor[2], pooling=pooling, gp=None, **conv_kwargs)

        cat_channels = COMB_FNS[comb_fn]['cat']
        comb_fn = COMB_FNS[comb_fn]['fn']

        # Build decoders (same number of feature maps as MALA)
        f2d = initial_num_fmaps * fmap_growth**2
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        decoders = [
            Up3D(f0b + f2e * cat_channels, f2d, 3, self.scale_factor[2], comb_fn=comb_fn, gp=None, **conv_kwargs),
            Up3D(f2d + f1e * cat_channels, f1d, 3, self.scale_factor[1], comb_fn=comb_fn, gp=None, **conv_kwargs),
            Up3D(f1d + f0e * cat_channels, f0d, 3, self.scale_factor[0], comb_fn=comb_fn, gp=gp, **conv_kwargs)
        ]

        # FIXME this is broken ?
        # Build output
        # output = Output(f0d, out_channels, 3)
        # multiply by 2 for real/imag concatenation
        gp_mult = conv_kwargs['no_g'] if gp is None else 1
        output = nn.Conv3d(f0d * 2 * gp_mult, out_channels, kernel_size=3, padding=1)

        # Parse final activation
        if final_activation == 'auto':
            self.final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()
        else:
            self.final_activation = getattr(nn, final_activation)()

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.base = base
        self.output = output

    def forward(self, input_):

        x = new_cmplx(input_)
        # print('INPUT')
        # print(f'x.size()={tuple(x.size())}')
        encoder_out = []
        # apply encoders and remember their outputs
        for encoder in self.encoders:
            x = encoder(x)
            encoder_out.append(x)
            # print('ENCODER')
            # print(f'x.size()={tuple(x.size())}')

        x = self.base(x)
        # print('BASE')
        # print(f'x.size()={tuple(x.size())}')

        # apply decoders
        max_level = len(self.decoders) - 1
        for level, decoder in enumerate(self.decoders):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(x, encoder_out[max_level - level])
            # print('DECODER')
            # print(f'x.size()={tuple(x.size())}')

        # from complex to real
        x = concatenate(x)
        x = x.view(x.size(0), -1, x.size(-3), x.size(-2), x.size(-1))
        x = self.output(x)

        # print('OUTPUT')
        # print(f'x.size()={tuple(x.size())}')
        if self.final_activation is not None:
            x = self.final_activation(x)
        # print('FINAL ACTIVATION')
        # print(f'x.size()={tuple(x.size())}')
        return x
