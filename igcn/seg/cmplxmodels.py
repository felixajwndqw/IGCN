import torch
import torch.nn as nn
from quicktorch.models import Model
from igcn.seg.cmplxigcn_unet_parts import DownCmplx, UpCmplx, TripleIGConvCmplx
from igcn.cmplx import new_cmplx


class UNetIGCNCmplx(Model):
    def __init__(self, n_classes, n_channels=1, no_g=4, base_channels=16,
                 kernel_size=3, mode='nearest', **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size

        self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size, no_g=no_g)
        self.down1 = DownCmplx(base_channels, base_channels * 2, kernel_size, no_g=no_g)
        self.down2 = DownCmplx(base_channels * 2, base_channels * 4, kernel_size, no_g=no_g)
        self.down3 = DownCmplx(base_channels * 4, base_channels * 8, kernel_size, no_g=no_g)
        self.down4 = DownCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g=no_g)
        self.up1 = UpCmplx(base_channels * 8, base_channels * 4, kernel_size, no_g=no_g, mode=mode)
        self.up2 = UpCmplx(base_channels * 4, base_channels * 2, kernel_size, no_g=no_g, mode=mode)
        self.up3 = UpCmplx(base_channels * 2, base_channels, kernel_size, no_g=no_g, mode=mode)
        self.up4 = UpCmplx(base_channels, base_channels, kernel_size, no_g=no_g, mode=mode, last=True)
        self.outc = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1)
        self.outc = nn.Conv2d(base_channels * 2, n_classes, kernel_size=1)

    def forward(self, x):
        x = new_cmplx(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x[0], x[1]], dim=1)
        mask = self.outc(x)
        return mask
