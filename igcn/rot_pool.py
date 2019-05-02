import torch.functional as F
from torch.nn.modules.pooling import _MaxPoolNd
import cv2


class RotMaxPool2d(_MaxPoolNd):
    def forward(self, input, thetas):
        thetas = thetas.repeat_interleave(input.size(0) // thetas.size(0))
        rot = [self.rotate(inp, th) for inp, th in zip(input, thetas)]
        rot_pool = F.max_pool2d(rot, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)
        return [self.rotate(rp, -th) for rp, th in zip(rot_pool, thetas)]

    def rotate(a, t):
        rows, cols = a.shape
        M = cv2.getRotationMatrix2D(((rows-1)/2, (cols-1)/2), t, 1)
        return cv2.warpAffine(a, M, (cols, rows))