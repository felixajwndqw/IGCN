import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from igcn.cmplx import cmplx
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx_modules import IGConvCmplx, IGConvGroupCmplx, Project, ReLUCmplx, MaxPoolCmplx, AvgPoolCmplx, MaxMagPoolCmplx, BatchNormCmplxOld, ReshapeGabor, GaborPool
from igcn.utils import _compress_shape, _recover_shape


class TripleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, include_gparams=False, last=False, gp='max',
                 relu_type='mod',
                 first=False, **kwargs):
        super().__init__()
        first_conv = IGConvGroupCmplx
        if first:
            first_conv = IGConvCmplx
        if type(kernel_size) is tuple:
            padding = kernel_size[-1] // 2
        else:
            padding = kernel_size // 2

        self.conv1 = first_conv(
            in_channels, out_channels, no_g=no_g,
            kernel_size=kernel_size,# if not first else 11,
            padding=padding,# if not first else 5,
            gabor_pooling=None, **kwargs)
        self.bn_relu1 = nn.Sequential(
            # BatchNormCmplx(out_channels),
            BatchNormCmplxOld(),
            ReLUCmplx(relu_type=relu_type, channels=out_channels, inplace=True)
        )

        self.conv2 = IGConvGroupCmplx(
            out_channels, out_channels, no_g=no_g,
            kernel_size=kernel_size,
            padding=padding, gabor_pooling=None, **kwargs)
        self.bn_relu2 = nn.Sequential(
            # BatchNormCmplx(out_channels),
            BatchNormCmplxOld(),
            ReLUCmplx(relu_type=relu_type, channels=out_channels, inplace=True)
        )

        self.conv3 = IGConvGroupCmplx(
            out_channels, out_channels, no_g=no_g,
            kernel_size=kernel_size,
            padding=padding, gabor_pooling=gp,
            include_gparams=include_gparams, **kwargs)
        self.bn_relu3 = nn.Sequential(
            # BatchNormCmplx(out_channels),
            BatchNormCmplxOld(),
            ReLUCmplx(relu_type=relu_type, channels=out_channels, inplace=True)
        )

        self.include_gparams = include_gparams

        # if include_gparams:
        #     self.handle_last = self.extract_theta
        # else:
        #     self.handle_last = self.normal_conv

    # def extract_theta(self, x):
    #     x, t = self.conv3(x)
    #     # print(f'x.size()={x.size()}, t.type()={t.type()}, self.include_gparams={self.include_gparams}')
    #     return self.bn_relu3(x), t

    # def normal_conv(self, x):
    #     return self.bn_relu3(self.conv3(x))

    def forward(self, x):
        # print(f'x.size()={x.size()}')
        x1 = self.bn_relu1(self.conv1(x))
        # print(f'x1.size()={x1.size()}')
        x2 = self.bn_relu2(self.conv2(x1))
        # print(f'x2.size()={x2.size()}')
        out = self.bn_relu3(self.conv3(x2 + x1))
        return out
        # return self.handle_last(x1 + x2)


    # def extract_theta(self):
    #     print(self.conv1[0].gabor.gabor_params[0])
    #     t = torch.mean(
    #         torch.stack((
    #             self.conv1[0].gabor.gabor_params[0],
    #             self.conv2[0].gabor.gabor_params[0],
    #             self.conv3[0].gabor.gabor_params[0]
    #         ))
    #     )
    #     return t


class DownCmplx(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, pooling='avg', gp='max', relu_type='mod', **kwargs):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(2)
        elif pooling == 'mag':
            Pool = MaxMagPoolCmplx(2)
        else:
            Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, gp=gp, relu_type=relu_type, **kwargs)
        )

    def forward(self, x):
        return self.pool_conv(x)


class DownCmplxAngle(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, pooling='avg', gp='max', relu_type='mod'):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(2)
        elif pooling == 'mag':
            Pool = MaxMagPoolCmplx(2)
        else:
            Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, gp=gp, include_gparams=True, relu_type=relu_type)
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

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, mode='nearest', last=False, gp=None, relu_type='mod', **kwargs):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, last=last, gp=gp, relu_type=relu_type, **kwargs)

    def forward(self, x1, x2):
        x1, xs = _compress_shape(x1)

        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))

        x1 = _recover_shape(x1, xs)

        return self.conv(x1 + x2)


class UpSimpleCmplx(nn.Module):
    """Upscaling then double conv

    Args:
        in_channels:
        out_channels
        mode (str, optional): Upsampling method. If None ConvTranspose2d will
            be used. Defaults to 'nearest'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, mode='nearest', last=False, gp=None):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, last=last, gp=gp)

    def forward(self, x1):
        x1, xs = _compress_shape(x1)

        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))

        x1 = _recover_shape(x1, xs)

        return self.conv(x1)


class RCFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, no_g=4,
                 layers=2, gp=None, relu_type='mod', **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                IGConvGroupCmplx(inc, outc, kernel_size, no_g=no_g, padding=1),
                ReLUCmplx(inplace=True, relu_type=relu_type, channels=outc)
            )
            for inc, outc
            in (
                (in_channels, out_channels),
                *((out_channels, out_channels),) * (layers - 1)
            )
        ])
        self.refines = nn.ModuleList([
            IGConvGroupCmplx(out_channels, 21, 1, no_g=no_g)
            for _ in range(layers)
        ])

        self.one_by = nn.Sequential(
            IGConvGroupCmplx(21, 1, 1, no_g=no_g),
            GaborPool(gp),
            ReshapeGabor(),
            Project('cat'),
            nn.Conv2d(2 * (1 if gp is not None else no_g), 1, 1)
        )

    def forward(self, x):
        side = torch.zeros(2, x.size(1), 21, *x.size()[3:], device=x.device)

        for conv, refine in zip(self.convs, self.refines):
            x = conv(x)
            side = side + refine(x)

        side = self.one_by(side)

        return x, side
