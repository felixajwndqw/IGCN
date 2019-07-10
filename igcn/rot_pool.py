import torchgeometry as tgm
from torch.nn.modules.pooling import _MaxPoolNd
import torch.nn.functional as F
import torch
import cv2


class RotMaxPool2d(_MaxPoolNd):
    def forward(self, input, thetas):
        rot = self.rotate(input, thetas)
        rot_pool = F.max_pool2d(rot, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)
        return self.rotate(rot_pool, -thetas)

    def rotate(self, input, thetas):
        rot = torch.empty_like(input)
        rot_batch = input.size(1) // thetas.size(0)
        _, _, rows, cols = input.size()
        centre = torch.tensor(((rows-1)/2, (cols-1)/2)).unsqueeze(0).repeat(input.size(0), 1).cuda()
        scale = torch.ones(1).repeat(input.size(0)).cuda()
        for i, th in enumerate(thetas.unsqueeze(1)):
            M = tgm.get_rotation_matrix2d(centre, th.repeat(input.size(0)), scale)
            rot[:, i * rot_batch: (i + 1) * rot_batch] = tgm.warp_affine(input[:, i * rot_batch: (i + 1) * rot_batch], M, dsize=(rows, cols))
        return rot
