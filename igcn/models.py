import torch
import torch.nn as nn
from quicktorch.models import Model
from igcn import IGConv
from igcn.cmplx_modules import IGConvCmplx, ReLUCmplx, BatchNormCmplx, MaxPoolCmplx, AvgPoolCmplx
from igcn.cmplx import new_cmplx


class DoubleIGConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling='max',
                 no_g=4, prev_max_gabor=False, max_gabor=False, first=False):
        super().__init__()
        padding = kernel_size // 2
        first_div = 2 if first else 1
        max_g_div = no_g if max_gabor else 1
        prev_max_g_div = no_g if max_gabor else 1
        if 'max' in pooling:
            Pool = MaxPoolCmplx
        elif pooling == 'avg':
            Pool = AvgPoolCmplx
        self.double_conv = nn.Sequential(
            IGConv(
                in_channels // prev_max_g_div,
                out_channels // first_div,
                kernel_size,
                pooling=Pool,
                padding=padding,
                no_g=no_g,
                max_gabor=False
            ),
            IGConv(
                out_channels // first_div,
                out_channels // max_g_div,
                kernel_size,
                pooling=Pool,
                padding=padding,
                no_g=no_g,
                max_gabor=max_gabor,
                pool_kernel=2,
                pool_stride=2
            ),
            ReLUCmplx(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, prev_max_gabor=False, max_gabor=False,
                 pooling='maxmag',
                 first=False, last=True):
        super().__init__()
        padding = kernel_size // 2 - 1
        first_div = 2 if first else 1
        max_g_div = no_g if max_gabor else 1
        prev_max_g_div = no_g if prev_max_gabor else 1
        if 'max' in pooling:
            Pool = MaxPoolCmplx
        elif pooling == 'avg':
            Pool = AvgPoolCmplx
        self.double_conv = nn.Sequential(
            IGConvCmplx(
                in_channels // prev_max_g_div,
                out_channels // first_div,
                kernel_size,
                padding=padding,
                no_g=no_g,
                max_gabor=False
            ),
            IGConvCmplx(
                out_channels // first_div,
                out_channels // max_g_div,
                kernel_size,
                padding=padding + int(last),
                no_g=no_g,
                max_gabor=max_gabor
            ),
            Pool(kernel_size=2, stride=2),
            BatchNormCmplx(),
            ReLUCmplx(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SingleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, prev_max_gabor=False, max_gabor=False,
                 pooling='maxmag',
                 last=True, **kwargs):
        super().__init__()
        print(f'out_channels={out_channels}')
        padding = kernel_size // 2 - 1
        max_g_div = no_g if max_gabor else 1
        prev_max_g_div = no_g if prev_max_gabor else 1
        if 'max' in pooling:
            Pool = MaxPoolCmplx
        elif pooling == 'avg':
            Pool = AvgPoolCmplx
        self.double_conv = nn.Sequential(
            IGConvCmplx(
                in_channels // prev_max_g_div,
                out_channels // max_g_div,
                kernel_size,
                padding=padding,
                no_g=no_g,
                max_gabor=max_gabor
            ),
            Pool(kernel_size=2, stride=2),
            BatchNormCmplx(),
            ReLUCmplx(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class IGCNNew(Model):
    def __init__(self, n_classes=10, n_channels=1, base_channels=16, no_g=4,
                 kernel_size=3, inter_mg=False, final_mg=False, cmplx=False,
                 pooling='max', dropout=0.3, dset='mnist', single=False):
        self.name = (f'igcn_{kernel_size}_{dset}_'
                     f'base_channels={base_channels}_'
                     f'no_g={no_g}_'
                     f'cmplx={cmplx}_'
                     f'inter_mg={inter_mg}_'
                     f'final_mg={final_mg}_')
        super(IGCNNew, self).__init__()
        if cmplx:
            ConvBlock = DoubleIGConvCmplx
            if single:
                ConvBlock = SingleIGConvCmplx
        else:
            ConvBlock = DoubleIGConv
        self.conv1 = ConvBlock(
            n_channels,
            base_channels * 2,
            kernel_size,
            no_g=no_g,
            max_gabor=inter_mg,
            pooling=pooling,
            first=True
        )
        self.conv2 = ConvBlock(
            base_channels * 2,
            base_channels * 3,
            kernel_size,
            no_g=no_g,
            prev_max_gabor=inter_mg,
            max_gabor=inter_mg,
            pooling=pooling
        )
        self.conv3 = ConvBlock(
            base_channels * 3,
            base_channels * 4,
            kernel_size,
            no_g=no_g,
            prev_max_gabor=inter_mg,
            max_gabor=final_mg,
            pooling=pooling,
            last=True
        )
        self.classifier = nn.Sequential(
            nn.Linear((2 if cmplx else 1) * 4 * base_channels // (no_g if final_mg else 1) * (4 if n_channels == 3 else 1), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 10)
        )
        self.cmplx = cmplx

    def forward(self, x):
        if self.cmplx:
            x = new_cmplx(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.cmplx:
            x = torch.cat([x[0], x[1]], dim=1)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class IGCN(Model):
    def __init__(self, no_g=4, model_name="default", rot_pool=None, dset="mnist",
                 inter_mg=False, final_mg=False, cmplx=False, one=True):
        self.name = (f'igcn_{model_name}_{dset}_'
                     f'no_g={no_g}_'
                     f'rot_pool={rot_pool}_'
                     f'inter_mg={inter_mg}_'
                     f'final_mg={final_mg}_')
        super(IGCN, self).__init__()
        self.create_feature_block(no_g, model_name, rot_pool, dset, inter_mg, final_mg)
        cmplx_mult = 2 if cmplx else 1
        out_size = 1 if one else 7
        self.classifier = nn.Sequential(
            nn.Linear(cmplx_mult * out_size * out_size * 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 10)
        )
        self.cmplx = cmplx

    def forward(self, x):
        if self.cmplx:
            x = new_cmplx(x)
        x = self.features(x)
        if self.cmplx:
            x = torch.cat([x[0], x[1]], dim=1)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def create_feature_block(self, no_g, model_name, rot_pool, dset, inter_mg, final_mg):
        modules = []
        if dset == "mnist" or dset == "mnistrot":
            if model_name == "default" or model_name == "3":
                modules = [
                    IGConv(1, 16, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=inter_mg, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(32, 48, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(48, 64, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=final_mg),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "5":
                modules = [
                    IGConv(1, 16, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=inter_mg, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(32, 48, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(48, 64, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=final_mg),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "7":
                modules = [
                    IGConv(1, 16, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=inter_mg, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(32, 48, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(48, 64, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=final_mg),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "9":
                modules = [
                    IGConv(1, 16, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=inter_mg, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(32, 48, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=inter_mg),
                    nn.ReLU(inplace=True),
                    IGConv(48, 64, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=final_mg),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "lp":
                modules = [
                    IGConv(1, 16, 9, rot_pool=None, padding=4, no_g=no_g, max_gabor=False),
                    IGConv(16, 32, 9, rot_pool=False, padding=3, no_g=no_g, max_gabor=False, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(32, 32, 9, rot_pool=None, padding=4, no_g=no_g, max_gabor=False),
                    IGConv(32, 48, 9, rot_pool=False, padding=4, no_g=no_g, max_gabor=False),
                    nn.ReLU(inplace=True),
                    IGConv(48, 64, 7, rot_pool=False, padding=4, no_g=no_g, max_gabor=True),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "3c":
                modules = [
                    IGConvCmplx(1, 16, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(16, 32, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 48, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 64, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "5c":
                modules = [
                    IGConvCmplx(1, 16, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(16, 32, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 48, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 64, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "7c":
                modules = [
                    IGConvCmplx(1, 16, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(16, 32, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 48, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 64, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "9c":
                modules = [
                    IGConvCmplx(1, 16, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(16, 32, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 48, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 64, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "lpc":
                modules = [
                    IGConvCmplx(1, 16, 7, rot_pool=None, padding=4, no_g=no_g, max_gabor=False),
                    IGConvCmplx(16, 32, 7, rot_pool=False, padding=4, no_g=no_g, max_gabor=False),
                    MaxPoolCmplx(kernel_size=3, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 32, 7, rot_pool=None, padding=4, no_g=no_g, max_gabor=False),
                    IGConvCmplx(32, 48, 7, rot_pool=False, padding=4, no_g=no_g, max_gabor=False),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 48, 7, rot_pool=False, padding=3, no_g=no_g, max_gabor=False),
                    IGConvCmplx(48, 64, 7, rot_pool=False, padding=3, no_g=no_g, max_gabor=True),
                    MaxPoolCmplx(kernel_size=3, stride=1),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "3o":
                modules = [
                    IGConv(1, 16, 3, rot_pool=None, padding=0, no_g=no_g, max_gabor=False),
                    IGConv(16, 32, 3, rot_pool=False, padding=0, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(32, 32, 3, rot_pool=None, padding=0, no_g=no_g, max_gabor=False),
                    IGConv(32, 48, 3, rot_pool=False, padding=0, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(48, 48, 3, rot_pool=None, padding=0, no_g=no_g, max_gabor=False),
                    IGConv(48, 64, 3, rot_pool=False, padding=1, no_g=no_g, max_gabor=final_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "5o":
                modules = [
                    IGConv(1, 16, 5, rot_pool=None, padding=1, no_g=no_g, max_gabor=False),
                    IGConv(16, 32, 5, rot_pool=False, padding=1, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(32, 32, 5, rot_pool=None, padding=1, no_g=no_g, max_gabor=False),
                    IGConv(32, 48, 5, rot_pool=False, padding=1, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(48, 48, 5, rot_pool=None, padding=1, no_g=no_g, max_gabor=False),
                    IGConv(48, 64, 5, rot_pool=False, padding=2, no_g=no_g, max_gabor=final_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "7o":
                modules = [
                    IGConv(1, 16, 7, rot_pool=None, padding=2, no_g=no_g, max_gabor=False),
                    IGConv(16, 32, 7, rot_pool=False, padding=2, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(32, 32, 7, rot_pool=None, padding=2, no_g=no_g, max_gabor=False),
                    IGConv(32, 48, 7, rot_pool=False, padding=2, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(48, 48, 7, rot_pool=None, padding=2, no_g=no_g, max_gabor=False),
                    IGConv(48, 64, 7, rot_pool=False, padding=3, no_g=no_g, max_gabor=final_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "9o":
                modules = [
                    IGConv(1, 16, 9, rot_pool=None, padding=3, no_g=no_g, max_gabor=False),
                    IGConv(16, 32, 9, rot_pool=False, padding=3, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(32, 32, 9, rot_pool=None, padding=3, no_g=no_g, max_gabor=False),
                    IGConv(32, 48, 9, rot_pool=False, padding=3, no_g=no_g, max_gabor=inter_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True),
                    IGConv(48, 48, 9, rot_pool=None, padding=3, no_g=no_g, max_gabor=False),
                    IGConv(48, 64, 9, rot_pool=False, padding=4, no_g=no_g, max_gabor=final_mg, pool_kernel=2, pool_stride=2),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "3oc":
                modules = [
                    IGConvCmplx(1, 16, 3, rot_pool=None, padding=0, no_g=no_g, max_gabor=False),
                    IGConvCmplx(16, 32, 3, rot_pool=False, padding=0, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 32, 3, rot_pool=None, padding=0, no_g=no_g, max_gabor=False),
                    IGConvCmplx(32, 48, 3, rot_pool=False, padding=0, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 48, 3, rot_pool=None, padding=0, no_g=no_g, max_gabor=False),
                    IGConvCmplx(48, 64, 3, rot_pool=False, padding=1, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "5oc":
                modules = [
                    IGConvCmplx(1, 16, 5, rot_pool=None, padding=1, no_g=no_g, max_gabor=False),
                    IGConvCmplx(16, 32, 5, rot_pool=False, padding=1, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 32, 5, rot_pool=None, padding=1, no_g=no_g, max_gabor=False),
                    IGConvCmplx(32, 48, 5, rot_pool=False, padding=1, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 48, 5, rot_pool=None, padding=1, no_g=no_g, max_gabor=False),
                    IGConvCmplx(48, 64, 5, rot_pool=False, padding=2, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "7oc":
                modules = [
                    IGConvCmplx(1, 16, 7, rot_pool=None, padding=2, no_g=no_g, max_gabor=False),
                    IGConvCmplx(16, 32, 7, rot_pool=False, padding=2, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 32, 7, rot_pool=None, padding=2, no_g=no_g, max_gabor=False),
                    IGConvCmplx(32, 48, 7, rot_pool=False, padding=2, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 48, 7, rot_pool=None, padding=2, no_g=no_g, max_gabor=False),
                    IGConvCmplx(48, 64, 7, rot_pool=False, padding=3, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
            if model_name == "9oc":
                modules = [
                    IGConvCmplx(1, 16, 9, rot_pool=None, padding=3, no_g=no_g, max_gabor=False),
                    IGConvCmplx(16, 32, 9, rot_pool=False, padding=3, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(32, 32, 9, rot_pool=None, padding=3, no_g=no_g, max_gabor=False),
                    IGConvCmplx(32, 48, 9, rot_pool=False, padding=3, no_g=no_g, max_gabor=inter_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True),
                    IGConvCmplx(48, 48, 9, rot_pool=None, padding=3, no_g=no_g, max_gabor=False),
                    IGConvCmplx(48, 64, 9, rot_pool=False, padding=4, no_g=no_g, max_gabor=final_mg),
                    MaxPoolCmplx(kernel_size=2, stride=2),
                    BatchNormCmplx(),
                    ReLUCmplx(inplace=True)
                ]
        if dset == "cifar":
            if model_name == "default" or model_name == "3":
                modules = [
                    IGConv(1, 8, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "5":
                modules = [
                    IGConv(1, 8, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "7":
                modules = [
                    IGConv(1, 8, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "9":
                modules = [
                    IGConv(1, 8, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
        self.features = nn.Sequential(*modules)