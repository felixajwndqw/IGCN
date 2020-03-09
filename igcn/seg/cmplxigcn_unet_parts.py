import torch.nn as nn
from igcn.cmplx import cmplx
from igcn.cmplx_modules import IGConvCmplx, ReLUCmplx, BatchNormCmplx, MaxPoolCmplx, AvgPoolCmplx


class TripleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, include_gparams=False, last=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Sequential(
            IGConvCmplx(in_channels, out_channels, no_g=no_g,
                        kernel_size=kernel_size,
                        padding=padding, gabor_pooling=None),
            BatchNormCmplx(out_channels),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )
        self.conv2 = nn.Sequential(
            IGConvCmplx(out_channels, out_channels, no_g=no_g,
                        kernel_size=kernel_size,
                        padding=padding, gabor_pooling=None),
            BatchNormCmplx(out_channels),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )
        self.conv3 = nn.Sequential(
            IGConvCmplx(out_channels, out_channels, no_g=no_g,
                        kernel_size=kernel_size,
                        padding=padding, gabor_pooling='max',
                        include_gparams=include_gparams),
            BatchNormCmplx(out_channels),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1 + x2)
        return x3


class DownCmplx(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, pooling='avg'):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(2)
        else:
            Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpCmplx(nn.Module):
    """Upscaling then double conv

    Args:
        in_channels:
        out_channels
        mode (str, optional): Upsampling method. If None ConvTranspose2d will
            be used. Defaults to 'nearest'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, mode='nearest', last=False):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, last=last)

    def forward(self, x1, x2):
        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))
        return self.conv(x1 + x2)
