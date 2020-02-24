import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init


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
    if operator == 'maxmag':
        return max_mag_pool(x, kernel_size, **kwargs)

    return cmplx(
        pool(x[0], kernel_size, **kwargs),
        pool(x[1], kernel_size, **kwargs)
    )


def max_mag_pool(x, kernel_size, **kwargs):
    """Computes max magnitude pooling on complex tensors.
    """
    r = magnitude(x)
    _, idxs = F.max_pool2d(r, kernel_size, return_indices=True, **kwargs)
    cmplx_idxs = cmplx(idxs, idxs)
    max_by_mags = x.flatten(start_dim=3).gather(dim=3, index=cmplx_idxs.flatten(start_dim=3))
    return max_by_mags.view_as(cmplx_idxs)


def init_weights(re, im, mode='he'):
    """Initialises conv. weights according to C. Trabelsi, Deep Complex Networks
    """
    assert(re.size() == im.size())
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(re)
    if mode == 'he':
        sigma = 1 / fan_in
    if mode == 'glorot':
        sigma = 1 / (fan_in + fan_out)

    mag = re.new_tensor(np.random.rayleigh(scale=sigma, size=re.size()))
    phase = re.new_tensor(np.random.uniform(low=-np.pi, high=np.pi, size=re.size()))

    with torch.no_grad():
        re.data = mag * torch.cos(phase)
        im.data = mag * torch.sin(phase)


    # # Delete this
    # exp = math.sqrt(math.pi / (2 * fan_in))
    # var = (4 - math.pi) / (2 * (fan_in))
    # varw = 2 / ((fan_in))
    # print(f"Expecting E(|W|)={exp}, Var(|W|)={var}, Var(W)={varw}")






# from torch.nn.modules.conv import Conv2d


# def main():
#     ReConv = Conv2d(4, 8, (3, 3))
#     ImConv = Conv2d(4, 8, (3, 3))
#     print(ReConv.weight.min(), ReConv.weight.max(), ReConv.weight.mean())
#     pre_init = magnitude(cmplx(ReConv.weight, ImConv.weight))
#     print(pre_init.mean().item())

#     init_weights(ReConv.weight, ImConv.weight)

#     post_init = magnitude(cmplx(ReConv.weight, ImConv.weight))
#     print(post_init.mean().item(), post_init.std().item() ** 2)
#     print(ReConv.weight.min(), ReConv.weight.max(), ReConv.weight.mean())


# if __name__ == '__main__':
#     main()
