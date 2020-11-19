# import torchgeometry as tgm
from torch.nn.modules.pooling import _MaxPoolNd
import torch.nn.functional as F
import torch
import time


class RotMaxPool2d(_MaxPoolNd):
    def forward(self, input, thetas):
        '''
        Performs rotated pooling on input.

        Args:
            input (torch.Tensor): Input data to rotate and pool.
            thetas (torch.Tensor): Angles to rotate by. Shape should be [N]
        '''
        pass
        # rot = self.rotate(input, thetas)
        # rot_pool = F.max_pool2d(rot, self.kernel_size, self.stride,
        #                         self.padding, self.dilation, self.ceil_mode,
        #                         self.return_indices)
        # out = self.rotate(rot_pool, -thetas)
        # return out

    def rotate(self, input, thetas):
        pass
        # rot = torch.empty_like(input)
        # rot_batch = input.size(1) // thetas.size(0)
        # _, _, rows, cols = input.size()
        # centre = rot.new_tensor(((rows-1)/2, (cols-1)/2)) \
        #     .unsqueeze(0).repeat(input.size(0), 1)
        # scale = rot.new_ones(input.size(0))
        # for i, th in enumerate(thetas.unsqueeze(1)):
        #     M = tgm.get_rotation_matrix2d(centre, th.repeat(input.size(0)), scale)
        #     rot[:, i * rot_batch: (i + 1) * rot_batch] = tgm.warp_affine(input[:, i * rot_batch: (i + 1) * rot_batch], M, dsize=(rows, cols))
        # return rot


def main():
    i_shape = (256, 1, 32, 32)
    t_shape = (8)
    # cpu_pooling = RotMaxPool2d(kernel_size=3, stride=1)

    # print("CPU START")
    # cpu_input = torch.randn(i_shape)
    # cpu_thetas = torch.randn(t_shape)
    # start = time.time()
    # rp_data = cpu_pooling(cpu_input, cpu_thetas)
    # print('Total CPU time: {}'.format(time.time() - start))

    print()
    print("GPU START")
    gpu_pooling = RotMaxPool2d(kernel_size=3, stride=1).cuda()
    gpu_input = torch.randn(i_shape).cuda()
    gpu_thetas = torch.randn(t_shape).cuda()
    start = time.time()
    for i in range(5):
        gpu_input = gpu_pooling(gpu_input, gpu_thetas)
    print('Total GPU time: {}'.format(time.time() - start))

if __name__ == '__main__':
    main()
