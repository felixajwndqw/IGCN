import torch


def _pair(x):
    if type(x) is int:
        return (x, x)
    return x
