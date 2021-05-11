import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from igcn.seg.cmplx_modules import DownCmplx, UpSimpleCmplx
from igcn.cmplx_modules import GaborPool, ReLUCmplx, IGConvGroupCmplx
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx import magnitude, cmplx
from igcn.utils import _compress_shape, _recover_shape


__all__ = ['PAM_Module', 'CAM_Module', 'GAM_Module', 'semanticModule']


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


class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplxTorch, gp='avg'):
        super(PAM_Module, self).__init__()
        self.query_conv = IGConvGroupCmplx(in_dim, max(in_dim//8, 1), kernel_size=1, no_g=no_g, gabor_pooling=gp)
        self.key_conv = IGConvGroupCmplx(in_dim, max(in_dim//8, 1), kernel_size=1, no_g=no_g, gabor_pooling=gp)
        self.value_conv = IGConvGroupCmplx(in_dim, in_dim, kernel_size=1, no_g=no_g)
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


# class CAM_Module(nn.Module):
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


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, no_g, Alignment=SoftmaxCmplxTorch, gp='avg'):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = Alignment(dim=-1)
        if gp is not None:
            self.gp = GaborPool(gp)
        else:
            self.gp = None

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
        proj_value = x.view(m_batchsize, C, -1)
        if self.gp is not None:
            x = self.gp(x)
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

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, G, height, width)

        out = self.gamma * out + x
        return cmplx(out.real, out.imag)


# class GAM_Module(nn.Module):
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


class GAM_Module(nn.Module):
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
        return cmplx(out.real, out.imag)


class AttentionLayer(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, no_g, AttentionModule=PAM_Module, Alignment=SoftmaxCmplxTorch, gp=None):
        super().__init__()

        self.attn = nn.Sequential(
            IGConvGroupCmplx(in_ch * 2, in_ch, kernel_size=3, padding=1, no_g=no_g),
            BatchNormCmplx(in_ch * no_g),
            ReLUCmplx(channels=in_ch),
            AttentionModule(in_ch, no_g, Alignment=Alignment, gp=gp),
            IGConvGroupCmplx(in_ch, in_ch, kernel_size=3, padding=1, no_g=no_g),
            BatchNormCmplx(in_ch * no_g),
            ReLUCmplx(channels=in_ch)
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
        super(MultiConv, self).__init__()

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
        # print(x.size(), xs)
        xs = [2, 4 * xs[1], *xs[2:]]
        x = cmplx(
            disassemble(x[0]),
            disassemble(x[1])
        )
        # print(x.size(), xs)
        x = _recover_shape(x, xs)
        # print(x.size(), xs)
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
