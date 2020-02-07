import torch
import torch.nn.functional as F


def cmplx(real, imag):
    """Stacks real and imaginary component into single tensor.

    This functions more as a readability aid than a helper method.
    """
    return torch.stack([real, imag], dim=0)


def new_cmplx(real):
    """Creates a trivial complex tensor.
    """
    return cmplx(real, torch.zeros_like(real))


def magnitude(x, eps=1e-8):
    """Computes magnitude of given complex tensor.

    Must return nonzero as grad(0)=inf
    """
    return torch.sqrt(torch.clamp(x.pow(2).sum(dim=0), min=eps))


def conv_cmplx(x, w, transpose=False, **kwargs):
    """Computes complex convolution.
    """
    conv = F.conv2d
    if transpose:
        conv = F.conv_transpose2d
        w = w.transpose(1, 2)

    real = conv(x[0, ...], w[0, ...], **kwargs) - conv(x[1, ...], w[1, ...], **kwargs)
    imag = conv(x[0, ...], w[1, ...], **kwargs) + conv(x[1, ...], w[0, ...], **kwargs)

    return cmplx(real, imag)


def relu_cmplx(x, inplace):
    """Computes complex relu.
    """
    r = magnitude(x)
    return F.relu(r) * x / r


def bnorm_cmplx(x, eps=1e-8):
    """Computes complex batch normalisation.
    """
    means = torch.mean(x, (1, 3, 4), keepdim=True)
    x = x - means

    stds = torch.std(magnitude(x, eps=eps), (0, 2, 3), keepdim=True)
    x = x / torch.clamp(stds.unsqueeze(0), min=eps)

    return x


def pool_cmplx(x, kernel_size, operator='max', **kwargs):
    """Computes complex pooling.
    """
    pool = F.max_pool2d
    if operator == 'avg' or operator == 'average':
        pool = F.avg_pool2d

    return cmplx(
        pool(x[0], kernel_size, **kwargs),
        pool(x[1], kernel_size, **kwargs)
    )
