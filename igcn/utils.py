import torch
import numpy as np


def _pair(x):
    if np.issubdtype(type(x), np.integer) and np.isscalar(x):
        return (x, x)
    return x


def _trio(x):
    if np.issubdtype(type(x), np.integer) and np.isscalar(x):
        return (x, x, x)
    return x


def _compress_shape(x):
    """Compresses gabor and feature channels into one axis
    """
    xs = None
    if x.dim() > 5:
        xs = x.size()
        x = x.view(
            2,
            xs[1],
            xs[2] * xs[3],
            *xs[4:]
        )

    return x, xs


def _recover_shape(x, xs):
    """Recovers gabor axis from original shape, while retaining spatial dims
    """
    if xs is not None:
        x = x.view(
            *xs[:4],
            *x.size()[3:]
        )

    return x
