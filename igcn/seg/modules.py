import torch
import torch.nn as nn
from igcn.modules import IGConv


class TripleIGConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, include_gparams=False, last=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Sequential(
            IGConv(in_channels, out_channels, no_g=no_g,
                   kernel_size=kernel_size,
                   padding=padding, max_gabor=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            IGConv(out_channels, out_channels, no_g=no_g,
                   kernel_size=kernel_size,
                   padding=padding, max_gabor=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            IGConv(out_channels, out_channels, no_g=no_g,
                   kernel_size=kernel_size,
                   padding=padding, max_gabor=last,
                   include_gparams=include_gparams),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1 + x2)
        return x3


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            TripleIGConv(in_channels, out_channels, kernel_size, no_g=no_g)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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

        self.conv = TripleIGConv(in_channels, out_channels, kernel_size, no_g=no_g, last=last)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(x1 + x2)


class RCFPlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 layers=2, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size, padding=1),
                nn.ReLU(inplace=True)
            )
            for inc, outc
            in (
                (in_channels, out_channels),
                *((out_channels, out_channels),) * (layers - 1)
            )
        ])
        self.refines = nn.ModuleList([
            nn.Conv2d(out_channels, 21, 1)
            for _ in range(layers)
        ])

        self.one_by = nn.Conv2d(21, 1, 1)

    def forward(self, x):
        side = torch.zeros(x.size(0), 21, *x.size()[2:], device=x.device)

        for conv, refine in zip(self.convs, self.refines):
            x = conv(x)
            side = side + refine(x)

        side = self.one_by(side)

        return x, side
