import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from igcn.cmplx import cmplx, bnorm_cmplx_old


class BatchNormCmplx(nn.Module):
    """Complex batch normalisation as described in Deep Complex Networks by Trabelsi et al.

    Implementation source: https://github.com/wavefrontshaping/complexPyTorch/.
    Modified a little for readability/compatibility.

    Args:
        num_features (int): Feature channels in.
        eps (float, optional): Epsilon buffer to prevent gradient explosion/vanishing.
            Defaults to 1e-5.
        momentum (float, optional): Coefficient for running mean/var weighted
            computation. If None cumulative moving average is used.
            Defaults to 1e-1.
        affine (bool, optional): Whether to use weights/biases.
            Defaults to True.
        track_running_stats (bool, optional): Whether to use running mean/var.
            Defaults to True.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, bnorm_type='new'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.bnorm_type = bnorm_type
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(2, num_features))
            self.register_buffer('running_covar', torch.zeros(3, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.reset_running_stats()
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[0, :].fill_(np.sqrt(2))
            self.running_covar[1, :].fill_(np.sqrt(2))
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], np.sqrt(2))
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)

    def forward(self, x):
        if self.bnorm_type == 'old':
            return bnorm_cmplx_old(x, self.eps)

        exponential_average_factor = 0.0
        xsh = x.size()
        x = x.view(2, xsh[1], xsh[2] * xsh[3], *xsh[4:])

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            # Shape is (RI, B, C * O, H, W)
            mean = x.mean((1, 3, 4))

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            x = x - mean[:, None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = x[0].numel() / x.size(2)
            Crr = 1. / n * x[0].pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * x[1].pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (x[0] * x[1]).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[0, :] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[0, :]

                self.running_covar[1, :] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[1, :]

                self.running_covar[2, :] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[2, :]

        else:
            mean = self.running_mean
            Crr = self.running_covar[0, :] + self.eps
            Cii = self.running_covar[1, :] + self.eps
            Cri = self.running_covar[2, :]  # +self.eps

            x = x - mean[:, None, :, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        x = cmplx(
            Rrr[None, :, None, None] * x[0] + Rri[None, :, None, None] * x[1],
            Rii[None, :, None, None] * x[1] + Rri[None, :, None, None] * x[0])

        if self.affine:
            x = cmplx(
                self.weight[None, :, 0, None, None] * x[0] +
                self.weight[None, :, 2, None, None] * x[1] +
                self.bias[None, :, 0, None, None],
                self.weight[None, :, 2, None, None] * x[0] +
                self.weight[None, :, 1, None, None] * x[1] +
                self.bias[None, :, 1, None, None])

        x = x.view(xsh)
        return x
