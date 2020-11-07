 __doc__ = "Model factory for IGCN.\n\n    Args:\n        n_classes (int, optional): Number of classes to estimate.\n        n_channels (int, optional): Number of input channels.\n        base_channels (int, optional): Number of feature channels in first layer of network.\n        no_g (int, optional): The number of desired Gabor filters.\n        kernel_size (int or tuple, optional): Size of kernel.\n        inter_gp (str, optional): Type of pooling to apply across Gabor\n            axis for intermediate layers.\n            Choices are [None, 'max', 'avg']. Defaults to None.\n        final_gp (str, optional): Type of pooling to apply across Gabor\n            axis for the final layer.\n            Choices are [None, 'max', 'avg']. Defaults to None.\n        cmplx (bool, optional): Whether to use a complex architecture.\n        pooling (str, optional): Type of pooling.\n        dropout (float, optional): Probability of dropout layer(s).\n        dset (str, optional): Type of dataset.\n        single (bool, optional): Whether to use a single gconv layer between each pooling layer.\n        all_gp (bool, optional): Whether to apply Gabor pooling on all layers.\n        relu_type (str, optional): Type of relu layer. Choices are ['c', 'z', 'mod'].\n            Defaults to 'c'.\n        nfc (int, optional): Number of fully connected layers before classification.\n        weight_init (str, optional): Type of weight initialisation.\n        fc_type (str, optional): How complex tensors are combined into real\n            tensors prior to FC. Choices are ['cat', 'mag']. Defaults to 'cat'.\n        fc_block(str, optional):\n        fc_relu_type(str, optional):\n        bnorm(str, optional):\n    "


import torch, torch.nn as nn
from quicktorch.models import Model
from igcn import IGConv
from igcn.cmplx_modules import IGConvCmplx, IGConvGroupCmplx, LinearCmplx, ReLUCmplx, MaxPoolCmplx, MaxMagPoolCmplx, AvgPoolCmplx, ConvCmplx, LinearMagPhase, Project, GaborPool
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx import new_cmplx, magnitude, concatenate

class DoubleIGConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pooling='max', no_g=4, prev_gabor_pooling=None, gabor_pooling=None, first=False):
        super().__init__()
        padding = kernel_size // 2
        first_div = 2 if first else 1
        max_g_div = no_g if gabor_pooling is not None else 1
        prev_max_g_div = no_g if gabor_pooling is not None else 1
        if 'max' in pooling:
            Pool = MaxPoolCmplx
        else:
            if pooling == 'avg':
                Pool = AvgPoolCmplx
        self.double_conv = nn.Sequential(IGConv((in_channels // prev_max_g_div),
          (out_channels // first_div),
          kernel_size,
          pooling=Pool,
          padding=padding,
          no_g=no_g,
          gabor_pooling=None), IGConv((out_channels // first_div),
          (out_channels // max_g_div),
          kernel_size,
          pooling=Pool,
          padding=padding,
          no_g=no_g,
          gabor_pooling=gabor_pooling,
          pool_kernel=2,
          pool_stride=2), ReLUCmplx(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class DoubleIGConvCmplx(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, no_g=4, prev_gabor_pooling=None, gabor_pooling=None, pooling='mag', weight_init=None, all_gp=False, relu_type='c', first=False, last=True, bnorm='new', group=False):
        super().__init__()
        padding = kernel_size // 2 - 1
        all_gp_div = no_g if all_gp else 1
        max_g_div = 1
        prev_max_g_div = 1
        first_div = 2 if first else 1
        all_gp = gabor_pooling if all_gp else None
        if group:
            Conv1 = Conv2 = IGConvGroupCmplx
            gabor_pooling = None
            all_gp = None
            if first:
                Conv1 = IGConvCmplx
        else:
            Conv1 = Conv2 = IGConvCmplx
        if pooling == 'max':
            Pool = MaxPoolCmplx
        else:
            if pooling == 'avg':
                Pool = AvgPoolCmplx
            else:
                if pooling == 'mag':
                    Pool = MaxMagPoolCmplx
        self.double_conv = nn.Sequential(Conv1((in_channels // prev_max_g_div),
          (out_channels // first_div // all_gp_div),
          kernel_size,
          padding=padding,
          no_g=no_g,
          gabor_pooling=all_gp,
          weight_init=weight_init), Conv2((out_channels // first_div // all_gp_div),
          (out_channels // max_g_div),
          kernel_size,
          padding=(padding + int(last)),
          no_g=no_g,
          gabor_pooling=gabor_pooling,
          weight_init=weight_init), Pool(kernel_size=2, stride=2), ReLUCmplx(inplace=True,
          relu_type=relu_type,
          channels=(out_channels // max_g_div)), BatchNormCmplx((out_channels // max_g_div), bnorm_type=bnorm))

    def forward(self, x):
        return self.double_conv(x)


class DoubleIGConvGroupCmplx(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, no_g=4, prev_gabor_pooling=None, gabor_pooling=None, pooling='mag', weight_init=None, all_gp=False, relu_type='c', first=False, last=False, bnorm='new', group=False):
        super().__init__()
        padding = kernel_size // 2 - 1
        first_div = 2 if first else 1
        Conv1 = Conv2 = IGConvGroupCmplx
        gp = None
        if last:
            gp = gabor_pooling
        if first:
            Conv1 = IGConvCmplx
        if pooling == 'max':
            Pool = MaxPoolCmplx
        else:
            if pooling == 'avg':
                Pool = AvgPoolCmplx
            else:
                if pooling == 'mag':
                    Pool = MaxMagPoolCmplx
        self.double_conv = nn.Sequential(Conv1(in_channels,
          (out_channels // first_div),
          kernel_size,
          padding=(padding + int(first)),
          no_g=no_g,
          weight_init=weight_init), Conv2((out_channels // first_div),
          out_channels,
          kernel_size,
          padding=(padding + int(first)),
          no_g=no_g,
          weight_init=weight_init,
          gabor_pooling=gp), Pool(kernel_size=2, stride=2), ReLUCmplx(inplace=True,
          relu_type=relu_type,
          channels=out_channels), BatchNormCmplx(out_channels, bnorm_type=bnorm))

    def forward(self, x):
        return self.double_conv(x)


class SingleIGConvCmplx(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, no_g=4, prev_gabor_pooling=None, gabor_pooling=None, pooling='mag', weight_init=None, last=True, **kwargs):
        super().__init__()
        padding = kernel_size // 2 - 1
        max_g_div = no_g if gabor_pooling is not None else 1
        prev_max_g_div = no_g if prev_gabor_pooling is not None else 1
        if pooling == 'max':
            Pool = MaxPoolCmplx
        else:
            if pooling == 'avg':
                Pool = AvgPoolCmplx
        if pooling == 'mag':
            Pool = MaxMagPoolCmplx
        self.double_conv = nn.Sequential(IGConvCmplx((in_channels // prev_max_g_div),
          (out_channels // max_g_div),
          kernel_size,
          padding=padding,
          no_g=no_g,
          gabor_pooling=gabor_pooling,
          weight_init=weight_init), Pool(kernel_size=2, stride=2), BatchNormCmplx(), ReLUCmplx(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class LinearBlock(nn.Module):

    def __init__(self, fcn, dropout=0.0, **kwargs):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(fcn, fcn), nn.ReLU(inplace=True), nn.Dropout(p=dropout))

    def forward(self, x):
        return self.block(x)


class LinearCmplxBlock(nn.Module):

    def __init__(self, fcn, dropout=0.0, relu_type='c', bias=True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(LinearCmplx(fcn, fcn, bias), ReLUCmplx(inplace=True, relu_type=relu_type, channels=fcn), nn.Dropout(p=dropout))

    def forward(self, x):
        return self.block(x)


class LinearConvBlock(nn.Module):

    def __init__(self, channels, dropout=0.0, relu_type='c', **kwargs):
        super().__init__()
        self.block = nn.Sequential(ConvCmplx(channels, channels, kernel_size=1), ReLUCmplx(inplace=True, relu_type=relu_type, channels=channels), nn.Dropout(p=dropout))

    def forward(self, x):
        return self.block(x)


class LinearMagPhaseBlock(nn.Module):

    def __init__(self, fcn, dropout=0.0, bias=True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(LinearMagPhase(fcn, fcn, bias), ReLUCmplx(inplace=True, relu_type='c', channels=fcn), nn.Dropout(p=dropout))

    def forward(self, x):
        return self.block(x)


class IGCN(Model):
    __doc__ = "Model factory for IGCN.\n\n    Args:\n        n_classes (int, optional): Number of classes to estimate.\n        n_channels (int, optional): Number of input channels.\n        base_channels (int, optional): Number of feature channels in first layer of network.\n        no_g (int, optional): The number of desired Gabor filters.\n        kernel_size (int or tuple, optional): Size of kernel.\n        inter_gp (str, optional): Type of pooling to apply across Gabor\n            axis for intermediate layers.\n            Choices are [None, 'max', 'avg']. Defaults to None.\n        final_gp (str, optional): Type of pooling to apply across Gabor\n            axis for the final layer.\n            Choices are [None, 'max', 'avg']. Defaults to None.\n        cmplx (bool, optional): Whether to use a complex architecture.\n        pooling (str, optional): Type of pooling.\n        dropout (float, optional): Probability of dropout layer(s).\n        dset (str, optional): Type of dataset.\n        single (bool, optional): Whether to use a single gconv layer between each pooling layer.\n        all_gp (bool, optional): Whether to apply Gabor pooling on all layers.\n        relu_type (str, optional): Type of relu layer. Choices are ['c', 'z', 'mod'].\n            Defaults to 'c'.\n        nfc (int, optional): Number of fully connected layers before classification.\n        weight_init (str, optional): Type of weight initialisation.\n        fc_type (str, optional): How complex tensors are combined into real\n            tensors prior to FC. Choices are ['cat', 'mag']. Defaults to 'cat'.\n        fc_block(str, optional):\n        fc_relu_type(str, optional):\n        bnorm(str, optional):\n    "

    def __init__(self, n_classes=10, n_channels=1, base_channels=16, no_g=4, kernel_size=3, inter_gp=None, final_gp=None, cmplx=False, pooling='max', dropout=0.3, dset='mnist', single=False, all_gp=False, relu_type='c', nfc=2, weight_init=None, fc_type='cat', fc_block='linear', fc_relu_type='c', bnorm='new', softmax=False, group=False, **kwargs):
        (super().__init__)(**kwargs)
        self.fc_type = fc_type
        if cmplx:
            ConvBlock = DoubleIGConvCmplx
            if single:
                ConvBlock = SingleIGConvCmplx
            if group:
                ConvBlock = DoubleIGConvGroupCmplx
        else:
            ConvBlock = DoubleIGConv
        self.fc_block = fc_block
        if fc_block == 'lin':
            FCBlock = LinearBlock
        else:
            if fc_block == 'clin':
                FCBlock = LinearCmplxBlock
            else:
                if fc_block == 'cnv':
                    FCBlock = LinearConvBlock
                else:
                    if 'mp' in fc_block:
                        FCBlock = LinearMagPhaseBlock
        self.conv1 = ConvBlock(n_channels,
          (base_channels * 2),
          kernel_size,
          no_g=no_g,
          gabor_pooling=inter_gp,
          pooling=pooling,
          first=True,
          weight_init=weight_init,
          all_gp=all_gp,
          relu_type=relu_type,
          bnorm=bnorm,
          group=group)
        self.conv2 = ConvBlock((base_channels * 2),
          (base_channels * 3),
          kernel_size,
          no_g=no_g,
          prev_gabor_pooling=inter_gp,
          gabor_pooling=inter_gp,
          pooling=pooling,
          weight_init=weight_init,
          all_gp=all_gp,
          relu_type=relu_type,
          bnorm=bnorm,
          group=group)
        self.conv3 = ConvBlock((base_channels * 3),
          (base_channels * 4),
          kernel_size,
          no_g=no_g,
          prev_gabor_pooling=inter_gp,
          gabor_pooling=final_gp,
          pooling=pooling,
          last=True,
          weight_init=weight_init,
          all_gp=all_gp,
          relu_type=relu_type,
          bnorm=bnorm,
          group=group)
        self.fcn = 4 * base_channels // (no_g if final_gp else 1) * (4 if n_channels == 3 else 1)
        if group:
            self.fcn = 4 * base_channels
        elif cmplx:
            if self.fc_block == 'lin':
                if self.fc_type == 'cat':
                    self.fcn *= 2
        else:
            linear_blocks = []
            for _ in range(nfc):
                linear_blocks.append(FCBlock((self.fcn), dropout, relu_type=fc_relu_type))

            self.linear = (nn.Sequential)(*linear_blocks)
            if self.fc_type == 'cat':
                if self.fc_block == 'cnv' or self.fc_block == 'clin' or 'mp' in self.fc_block:
                    self.fcn *= 2
            self.project = Project(self.fc_type)
            if self.fc_block == 'clin':
                if self.fc_type == 'cat':
                    self.classifier = nn.Sequential(self.project, nn.Linear(self.fcn, 10))
                else:
                    if self.fc_type == 'mag':
                        self.classifier = nn.Sequential(LinearCmplx(self.fcn, 10), self.project)
            else:
                if 'mp' in self.fc_block:
                    self.project = Project('mp')
                    self.classifier = nn.Sequential(Project('cat'), nn.Linear(self.fcn, 10))
                else:
                    if softmax:
                        self.classifier = nn.Sequential(nn.Linear(self.fcn, 10), nn.Softmax(dim=1))
                    else:
                        self.classifier = nn.Sequential(nn.Linear(self.fcn, 10))
        self.cmplx = cmplx

    def forward(self, x):
        if self.cmplx:
            x = new_cmplx(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.fc_block == 'cnv':
            x = self.linear(x)
            if self.cmplx:
                x = self.project(x)
            x = x.flatten(1)
            x = self.classifier(x)
        else:
            if self.fc_block == 'clin':
                x = x.flatten(2)
                x = self.linear(x)
                x = self.classifier(x)
        if 'mp' in self.fc_block:
            x = x.flatten(2)
            x = self.project(x)
            x = self.linear(x)
            x = self.classifier(x)
        else:
            if self.cmplx:
                x = self.project(x)
            x = x.flatten(1)
            x = self.linear(x)
            x = self.classifier(x)
        return x


class SFC(Model):

    def __init__(self, n_classes=10, n_channels=1, no_g=4, pooling='max', bnorm='old', final_gp='max', dropout=0.3, **kwargs):
        (super().__init__)(**kwargs)
        if pooling == 'max':
            Pool = MaxPoolCmplx
        else:
            if pooling == 'avg':
                Pool = AvgPoolCmplx
            else:
                if pooling == 'mag':
                    Pool = MaxMagPoolCmplx
        self.block1 = nn.Sequential(IGConvCmplx(n_channels, 24, 9, no_g, padding=2, weight_init='he'), ReLUCmplx(inplace=True), BatchNormCmplx((24 * no_g), bnorm_type=bnorm))
        self.block2 = nn.Sequential(IGConvGroupCmplx(24, 32, 7, no_g, padding=3, weight_init='he'), ReLUCmplx(inplace=True), BatchNormCmplx((32 * no_g), bnorm_type=bnorm))
        self.block3 = nn.Sequential(IGConvGroupCmplx(32, 36, 7, no_g, padding=3, weight_init='he'), ReLUCmplx(inplace=True), BatchNormCmplx((36 * no_g), bnorm_type=bnorm))
        self.block4 = nn.Sequential(IGConvGroupCmplx(36, 36, 7, no_g, padding=3, weight_init='he'), ReLUCmplx(inplace=True), BatchNormCmplx((36 * no_g), bnorm_type=bnorm))
        self.block5 = nn.Sequential(IGConvGroupCmplx(36, 64, 7, no_g, padding=2, weight_init='he'), ReLUCmplx(inplace=True), BatchNormCmplx((64 * no_g), bnorm_type=bnorm))
        self.block6 = nn.Sequential(IGConvGroupCmplx(64, 96, 5, no_g, padding=1, weight_init='he'), ReLUCmplx(inplace=True), BatchNormCmplx((96 * no_g), bnorm_type=bnorm))
        self.pool1 = Pool(kernel_size=2, stride=2)
        self.pool2 = Pool(kernel_size=2, stride=2)
        self.pool3 = Pool(kernel_size=2, stride=1)
        self.gpool = GaborPool(final_gp)
        self.project = Project('cat')
        fcn = 96 * self.project.mult
        self.linear = nn.Sequential(nn.Linear(fcn, fcn), nn.BatchNorm1d(fcn), nn.ELU(inplace=True), nn.Dropout(p=dropout), nn.Linear(fcn, fcn), nn.BatchNorm1d(fcn), nn.ELU(inplace=True), nn.Dropout(p=dropout))
        self.classifier = nn.Sequential(nn.Linear(fcn, n_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x = new_cmplx(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.gpool(x)
        x = self.project(x)
        x = self.linear(x.flatten(1))
        x = self.classifier(x)
        return x