import torch
import torch.nn as nn
from quicktorch.models import Model
from igcn.seg.igcn_unet_parts import Down, Up, TripleIGConv


class UNetIGCN(Model):
    def __init__(self, n_classes, n_channels=1, no_g=4,
                 kernel_size=3, mode='nearest', **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size

        self.inc = TripleIGConv(n_channels, 16, kernel_size, no_g=no_g)
        self.down1 = Down(16, 32, kernel_size, no_g=no_g)
        self.down2 = Down(32, 48, kernel_size, no_g=no_g)
        self.down3 = Down(48, 64, kernel_size, no_g=no_g)
        self.down4 = Down(64, 64, kernel_size, no_g=no_g)
        self.up1 = Up(64, 48, kernel_size, no_g=no_g, mode=mode)
        self.up2 = Up(48, 32, kernel_size, no_g=no_g, mode=mode)
        self.up3 = Up(32, 16, kernel_size, no_g=no_g, mode=mode)
        self.up4 = Up(16, 16, kernel_size, no_g=no_g, mode=mode, last=True)
        self.outc = nn.Conv2d(16, 16, kernel_size=1)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        mask = self.outc(x)
        return mask
