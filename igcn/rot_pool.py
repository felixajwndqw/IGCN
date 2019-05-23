import torchgeometry as tgm
from torch.nn.modules.pooling import _MaxPoolNd
import torch
import cv2


class RotMaxPool2d(_MaxPoolNd):
    def forward(self, input, thetas):
        # print("Initial thetas size:", thetas.size())
        # print("Output features is", input.size(1) // thetas.size(0))
        # thetas = thetas.repeat_interleave(input.size(1) // thetas.size(0))
        # print("After thetas size:", thetas.size())
        # for inp, th in zip(input, thetas):
        #     print("Individual size:", inp.size(), th.size())
        # rot = torch.tensor([[TF.rotate(ch, th) for ch, th in zip(channels, thetas)]
        #                     for channels in input])
        
        rot = self.rotate(input, thetas)
        rot_pool = F.max_pool2d(rot, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)
        return self.rotate(rot_pool, -thetas)

    def rotate(self, input, thetas):
        rot = torch.empty_like(input)
        rot_batch = input.size(1) // thetas.size(0)
        _, _, rows, cols = input.size()
        centre = torch.tensor(((rows-1)/2, (cols-1)/2)).unsqueeze(0).cuda()
        scale = torch.ones(1).cuda()
        print(thetas.size())
        for i, th in enumerate(thetas.unsqueeze(1)):
            print(centre.size(), th.size(), scale.size())
            M = tgm.get_rotation_matrix2d(centre, th, scale)
            rot[:, i*rot_batch:(i+1)*rot_batch] = tgm.warp_affine(input[:, i*rot_batch:(i+1)*rot_batch], M, dsize=(rows, cols))
        return rot
        # rows, cols = a.shape
        # M = cv2.getRotationMatrix2D(((rows-1)/2, (cols-1)/2), t, 1)
        # return torchgeometry.warp_affine(a, M, (cols, rows))