import math
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

from igcn.seg.cmplx_modules import DownCmplx, UpSimpleCmplx
from igcn.cmplx_modules import GaborPool, Project, ReLUCmplx, IGConvGroupCmplx, IGConvCmplx
from igcn.modules import IGConv
from igcn.modules import GaborPool as GaborPoolReal
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx import magnitude, cmplx, new_cmplx
from igcn.utils import _compress_shape, _recover_shape

from quicktorch.modules.attention.attention import (
    PositionAttentionHead,
    ChannelAttentionHead,
    DualAttentionHead,
    AttentionLayer,
    PAM_Module,
    CAM_Module
)


__all__ = ['PAMCmplx_Module', 'CAMCmplx_Module', 'GAMCmplx_Module', 'semanticModule']


def get_gabor_attention_head(key):
    return getattr(sys.modules[__name__], f'Gabor{key}AttentionHead')


class GaborDualAttentionHead(DualAttentionHead):
    """
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.pam = AttentionLayerGabor(channels, 4, PAMReal_Module)
        self.cam = AttentionLayerGabor(channels, 4, CAMReal_Module)


class GaborPositionAttentionHead(PositionAttentionHead):
    """
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.pam = AttentionLayerGabor(channels, 4, PAMReal_Module)


class GaborChannelAttentionHead(ChannelAttentionHead):
    """
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.cam = AttentionLayerGabor(channels, 4, CAMReal_Module)


class GaborDualCmplxAttentionHead(DualAttentionHead):
    """
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.pam = AttentionLayerGaborCmplx(channels, 4, PAMCmplx_Module)
        self.cam = AttentionLayerGaborCmplx(channels, 4, CAMCmplx_Module)


class GaborPositionCmplxAttentionHead(PositionAttentionHead):
    """
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.pam = AttentionLayerGaborCmplx(channels, 4, PAMCmplx_Module)


class GaborChannelCmplxAttentionHead(ChannelAttentionHead):
    """
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.cam = AttentionLayerGaborCmplx(channels, 4, CAMCmplx_Module)


class GaborTriAttentionHead(DualAttentionHead):
    def __init__(self, channels):
        super().__init__(channels)
        self.gam = AttentionLayerGabor(channels, 4, GAMReal_Module)

    def forward(self, fused, semantic):
        return self.conv(
            (
                self.pam(fused) +
                self.cam(fused) +
                self.gam(fused)
            ) *
            self.conv_semantic(semantic)
        )


class GaborTriCmplxAttentionHead(GaborTriAttentionHead):
    def __init__(self, channels):
        super().__init__(channels)
        self.gam = AttentionLayerGaborCmplx(channels, 4, GAMCmplx_Module)


class GaborTriGaborAttentionHead(GaborTriAttentionHead):
    def __init__(self, channels):
        super().__init__(channels)
        self.pam = AttentionLayerGabor(channels, 4, PAMReal_Module)
        self.cam = AttentionLayerGabor(channels, 4, CAMReal_Module)
        self.gam = AttentionLayerGabor(channels, 4, GAMReal_Module)


class GaborTriGaborCmplxAttentionHead(GaborTriAttentionHead):
    def __init__(self, channels):
        super().__init__(channels)
        self.pam = AttentionLayerGaborCmplx(channels, 4, PAMCmplx_Module)
        self.cam = AttentionLayerGaborCmplx(channels, 4, CAMCmplx_Module)
        self.gam = AttentionLayerGaborCmplx(channels, 4, GAMCmplx_Module)


class SoftmaxCmplx(nn.Module):
    """
    """
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(magnitude(x, sq=True), dim=self.dim)


class SoftmaxCmplxTorch(nn.Module):
    """
    """
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x.real ** 2 + x.imag ** 2, dim=self.dim)
        return torch.complex(
            F.softmax(x.real ** 2 + x.imag ** 2, dim=self.dim),
            torch.zeros_like(x, dtype=torch.float)
        )


class SemanticModule(nn.Module):
    """
    Semantic attention module
    """
    def __init__(self, in_dim, no_g):
        super().__init__()
        self.chanel_in = in_dim

        self.enc1 = DownCmplx(in_dim, in_dim*2, kernel_size=3, pooling='max', no_g=no_g, gp=None)
        self.enc2 = DownCmplx(in_dim*2, in_dim*4, kernel_size=3, pooling='max', no_g=no_g, gp=None)
        self.dec2 = UpSimpleCmplx(in_dim * 4, in_dim * 2, kernel_size=3, mode='bilinear', no_g=no_g)
        self.dec1 = UpSimpleCmplx(in_dim * 2, in_dim, kernel_size=3, mode='bilinear', no_g=no_g)

    def forward(self, x):
        x = self.enc1(x)
        enc = self.enc2(x)

        x = self.dec2(enc)
        x = self.dec1(x)

        return enc.view(-1), x


class PAMCmplx_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplxTorch, gp='avg'):
        super().__init__()
        self.query_conv = IGConvCmplx(in_dim * no_g, max(in_dim//8, 1), kernel_size=1, no_g=no_g, gabor_pooling=gp)
        self.key_conv = IGConvCmplx(in_dim * no_g, max(in_dim//8, 1), kernel_size=1, no_g=no_g, gabor_pooling=gp)
        self.value_conv = IGConvCmplx(in_dim * no_g, in_dim, kernel_size=1, no_g=no_g)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = Alignment(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps (2 X B X C X G X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        _, m_batchsize, C, G, height, width = x.size()
        # (WxH)x(1xC/8)
        proj_query = self.query_conv(x).view(2, m_batchsize, -1, width*height).permute(0, 1, 3, 2)
        proj_query = torch.complex(proj_query[0], proj_query[1])
        # (1xC/8)x(WxH)
        proj_key = self.key_conv(x).view(2, m_batchsize, -1, width*height)
        proj_key = torch.complex(proj_key[0], proj_key[1])

        # (WxH)x(WxH)
        energy = torch.bmm(proj_query, proj_key)

        attention = self.align(energy)
        proj_value = self.value_conv(x).view(2, m_batchsize, -1, width*height)
        # proj_value = torch.complex(proj_value[0], proj_value[1])

        out = cmplx(
            torch.bmm(proj_value[0], attention.permute(0, 2, 1)),
            torch.bmm(proj_value[1], attention.permute(0, 2, 1))
        )
        out = out.view(2, m_batchsize, C, G, height, width)
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, G, height, width)

        # out = cmplx(out.real, out.imag)
        out = self.gamma * out + x
        return out


class CAMCmplx_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplxTorch, gp='avg'):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = Alignment(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps (2 X B X C X G X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x = torch.complex(x[0], x[1])
        m_batchsize, C, G, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        # try cutting this op and see results
        # also consider using max mag i.e.
        # (torch.max(energy.abs(), -1, keepdim=True)[0].expand_as(energy) - energy.abs()) * 
        # torch.exp(torch.complex(torch.zeros_like(energy, dtype=torch.float), energy.angle()))
        energy = torch.complex(
            torch.max(energy.real, -1, keepdim=True)[0].expand_as(energy),
            torch.max(energy.imag, -1, keepdim=True)[0].expand_as(energy)
        ) - energy
        attention = self.align(energy)
        attention = torch.complex(
            attention,
            torch.zeros_like(attention, dtype=torch.float)
        )

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, G, height, width)

        out = self.gamma * out + x
        out = cmplx(out.real, out.imag)
        return out


class GAMCmplx_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplxTorch, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = Alignment(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps (2 X B X C X G X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x = torch.complex(x[0], x[1])
        m_batchsize, C, G, height, width = x.size()

        # Gx(CxWxH)
        proj_query = x.view(m_batchsize, G, -1)
        # (CxWxH)xG
        proj_key = x.view(m_batchsize, G, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.complex(
            torch.max(energy.real, -1, keepdim=True)[0].expand_as(energy),
            torch.max(energy.imag, -1, keepdim=True)[0].expand_as(energy)
        ) - energy
        attention = self.align(energy)
        attention = torch.complex(
            attention,
            torch.zeros_like(attention, dtype=torch.float)
        )
        proj_value = x.view(m_batchsize, G, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, G, height, width)

        out = self.gamma * out + x
        out = cmplx(out.real, out.imag)
        return out


class PAMReal_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, no_g, gp='max'):
        super().__init__()
        self.query_conv = IGConv(in_channels * no_g, max(in_channels//8, no_g), kernel_size=1, no_g=no_g, max_gabor=True)
        self.key_conv = IGConv(in_channels * no_g, max(in_channels//8, no_g), kernel_size=1, no_g=no_g, max_gabor=True)
        self.value_conv = IGConv(in_channels * no_g, in_channels * no_g, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)
        self.no_g = no_g

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X G X H X W)
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        C = C // self.no_g
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.align(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C * self.no_g, height, width)

        out = self.gamma * out + x
        return out


class CAMReal_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, no_g, gp='avg'):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)
        self.no_g = no_g

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X G X H X W)
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        C = C // self.no_g
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.align(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C * self.no_g, height, width)

        out = self.gamma * out + x
        return out


class GAMReal_Module(nn.Module):
    """ Gabor attention module"""
    def __init__(self, in_channels, no_g, gp='max'):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)
        self.no_g = no_g

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X G X H X W)
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        C = C // self.no_g
        proj_query = x.view(m_batchsize, self.no_g, -1)
        proj_key = x.view(m_batchsize, self.no_g, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.align(energy)
        proj_value = x.view(m_batchsize, self.no_g, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C * self.no_g, height, width)

        out = self.gamma * out + x
        return out


# class CAMCmplx_Module(nn.Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplx, **kwargs):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.align = Alignment(dim=-1)

#     def forward(self, x):
#         """
#         Parameters:
#         ----------
#             inputs :
#                 x : input feature maps (2 X B X C X G X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         _, m_batchsize, C, G, height, width = x.size()
#         # Cx(GxWxH)
#         proj_query = x.view(2, m_batchsize, C, -1)
#         proj_query = torch.complex(proj_query[0], proj_query[1])
#         # (GxWxH)xC
#         proj_key = x.view(2, m_batchsize, C, -1).permute(0, 1, 3, 2)
#         proj_key = torch.complex(proj_key[0], proj_key[1])

#         # CxC
#         energy = torch.bmm(proj_query, proj_key)
#         proj_query = None
#         proj_key = None

#         energy = cmplx(energy.real, energy.imag)
#         energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

#         attention = self.align(energy)
#         energy = None
#         attention = torch.complex(attention, torch.zeros(*attention.size(), device=attention.device))
#         proj_value = x.view(2, m_batchsize, C, -1)
#         proj_value = torch.complex(proj_value[0], proj_value[1])

#         out = torch.bmm(attention, proj_value)
#         proj_value = None
#         attention = None
#         out = cmplx(out.real, out.imag)
#         out = out.view(2, m_batchsize, C, G, height, width)

#         out = self.gamma * out + x
#         return out


# class GAMCmplx_Module(nn.Module):
#     """ Gabor orientation attention module"""
#     def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplx, **kwargs):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.align = Alignment(dim=-1)

#     def forward(self, x):
#         """
#         Parameters:
#         ----------
#             inputs :
#                 x : input feature maps (2 X B X C X G X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         _, m_batchsize, C, G, height, width = x.size()
#         # Gx(CxWxH)
#         proj_query = x.view(2, m_batchsize, G, -1)
#         proj_query = torch.complex(proj_query[0], proj_query[1])
#         # (CxWxH)xG
#         proj_key = x.view(2, m_batchsize, G, -1).permute(0, 1, 3, 2)
#         proj_key = torch.complex(proj_key[0], proj_key[1])

#         # GxG
#         energy = torch.bmm(proj_query, proj_key)
#         proj_query = None
#         proj_key = None

#         energy = cmplx(energy.real, energy.imag)
#         energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

#         attention = self.align(energy)
#         energy = None
#         attention = torch.complex(attention, torch.zeros(*attention.size(), device=attention.device))
#         proj_value = x.view(2, m_batchsize, G, -1)
#         proj_value = torch.complex(proj_value[0], proj_value[1])

#         out = torch.bmm(attention, proj_value)
#         proj_value = None
#         attention = None
#         out = cmplx(out.real, out.imag)
#         out = out.view(2, m_batchsize, C, G, height, width)

#         out = self.gamma * out + x
#         return out


class AttentionLayerGaborCmplx(nn.Module):
    """
    Helper function for complex-valued gabor attention modules

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAMCmplx_Module or CAMCmplx_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, no_g, AttentionModule=PAMCmplx_Module, Alignment=SoftmaxCmplxTorch, gp='max', project='cat'):
        super().__init__()

        cmplx_to_real = Project(project)
        self.attn = nn.Sequential(
            IGConvCmplx(in_ch * 2, in_ch, kernel_size=3, padding=1, no_g=no_g),
            BatchNormCmplx(in_ch * no_g),
            ReLUCmplx(channels=in_ch),
            AttentionModule(in_ch, no_g, Alignment=Alignment, gp=gp),
            GaborPool(gp),
            cmplx_to_real,
            nn.Conv2d(in_ch * cmplx_to_real.mult, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        x = new_cmplx(x)
        return self.attn(x)


class AttentionLayerGabor(nn.Module):
    """
    Helper Function for real-valued gabor attention modules

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAMCmplx_Module or CAMCmplx_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, no_g, AttentionModule=PAMReal_Module, Alignment=SoftmaxCmplxTorch, gp=None):
        super().__init__()

        self.attn = nn.Sequential(
            IGConv(in_ch * 2, in_ch * no_g, kernel_size=3, padding=1, no_g=no_g),
            nn.PReLU(),
            AttentionModule(in_ch, no_g, gp=gp),
            GaborPoolReal(no_g, gp),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)


class MultiConv(nn.Module):
    """
    Helper function for Multiple Convolutions for refining.

    Parameters:
    ----------
    inputs:
        in_ch : input channels
        out_ch : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """
    def __init__(self, in_ch, out_ch, no_g, attn=True):
        super().__init__()

        self.fuse_attn = nn.Sequential(
            IGConvGroupCmplx(in_ch, out_ch, kernel_size=3, padding=1, no_g=no_g),
            BatchNormCmplx(out_ch * no_g),
            ReLUCmplx(channels=out_ch),
            # IGConvGroupCmplx(out_ch, out_ch, kernel_size=3, padding=1, no_g=no_g),
            # BatchNormCmplx(out_ch * no_g),
            # ReLUCmplx(channels=out_ch),
            IGConvGroupCmplx(out_ch, out_ch, kernel_size=1, no_g=no_g),
            BatchNormCmplx(out_ch * no_g),
            nn.Softmax2d() if attn else ReLUCmplx()
        )

    def forward(self, x):
        return self.fuse_attn(x)


class Disassemble(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, xs = compress(x)
        x = disassemble(x)
        x = recover(x, xs)
        return x


class Reassemble(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, xs = compress(x)
        if xs is not None:
            xs = [xs[0] // 4, *xs[1:]]
        x = reassemble(x)
        x = recover(x, xs)
        return x


class DisassembleCmplx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, xs = _compress_shape(x)
        xs = [2, 4 * xs[1], *xs[2:]]
        x = cmplx(
            disassemble(x[0]),
            disassemble(x[1])
        )
        x = _recover_shape(x, xs)
        return x


class ReassembleCmplx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, xs = _compress_shape(x)
        if xs is not None:
            xs = [2, xs[1] // 4, *xs[2:]]
        x = cmplx(
            reassemble(x[0]),
            reassemble(x[1])
        )
        x = _recover_shape(x, xs)
        return x


def disassemble(x):
    _, c, w, h = x.size()
    x = x.unfold(2, w // 2, w // 2)
    x = x.unfold(3, h // 2, h // 2)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, c, w // 2, h // 2)
    return x


def reassemble(x):
    b, c, w, h = x.size()
    x = x.view(b // 4, 4, c, w, h)
    x = x.permute(0, 2, 3, 4, 1)
    x = x.reshape(b // 4, c * w * h, 4)
    x = F.fold(x, (w * 2, h * 2), (w, h), (1, 1), stride=(w, h))
    return x


def compress(x):
    xs = None
    if x.dim() == 5:
        xs = x.size()
        x = x.view(
            xs[0],
            xs[1] * xs[2],
            *xs[3:]
        )

    return x, xs


def recover(x, xs):
    if xs is not None:
        # x = x.view(xs)
        x = x.view(
            -1,
            *xs[1:3],
            x.size(-2),
            x.size(-1)
        )

    return x
