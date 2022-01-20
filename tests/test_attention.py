import torch
import igcn.seg.attention.attention as att
from quicktorch.modules.attention.attention import (
    GuidedAttention, SemanticModule
)


def test_att_heads():
    att_heads = [
        "Dual",
        "Position",
        "Channel",
        "Tri",
        "TriGabor",
    ]
    for att_head in att_heads:
        module = att.get_gabor_attention_head(att_head)(16)
        t = torch.arange(288).view(1, 32, 3, 3).to(torch.float32)
        fused = torch.arange(288, 576).view(1, 32, 3, 3).to(torch.float32)
        out = module(t, fused)
        assert out.shape == (1, 16, 3, 3), f"Incorrect output shape for {att_head}"


def test_att_heads_cmplx():
    att_heads = [
        "DualCmplx",
        "PositionCmplx",
        "ChannelCmplx",
        "TriCmplx",
        "TriGaborCmplx",
    ]
    for att_head in att_heads:
        head = att.get_gabor_attention_head(att_head)(16)
        t = torch.arange(288).view(1, 32, 3, 3).to(torch.float32)
        fused = torch.arange(288, 576).view(1, 32, 3, 3).to(torch.float32)
        out = head(t, fused)
        assert out.shape == (1, 16, 3, 3), f"Incorrect output shape for {att_head}"


def test_att_mod():
    att_heads = [
        "Dual",
        "Position",
        "Channel",
        "Tri",
        "TriGabor",
        "DualCmplx",
        "PositionCmplx",
        "ChannelCmplx",
        "TriCmplx",
        "TriGaborCmplx",
    ]
    for att_head in att_heads:
        head = att.get_gabor_attention_head(att_head)
        module = GuidedAttention(16, 0, SemanticModule(16 * 2), SemanticModule(16 * 2), head)
        t = torch.rand(1, 16, 128, 128).to(torch.float32)
        fused = torch.rand(1, 16, 128, 128).to(torch.float32)
        out, aux = module(t, fused)
        assert out.shape == (1, 16, 128, 128), f"Incorrect output shape for {att_head}"
