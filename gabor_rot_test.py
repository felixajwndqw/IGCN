from igcn.gabor import gabor_cmplx
import torch
import math
import matplotlib.pyplot as plt


def cmplx_mult(a, b):
    return torch.stack([
        a[0] * b[0] - a[1] * b[1],
        a[0] * b[1] + a[1] * b[0]
    ])


a = torch.ones(2, 1, 1, 10, 10)

no_g = 2

theta = torch.Tensor(no_g)
theta.data = torch.arange(no_g) / (no_g) * math.pi

lam = torch.Tensor(no_g)
lam.data.uniform_(
    -1 / math.sqrt(no_g),
    1 / math.sqrt(no_g)
)

g_params = torch.stack((
    theta,
    lam
))

g = gabor_cmplx(a, g_params)

print(a.size(), g.size())

mod_filter = a * g

print(mod_filter.size())

angle = torch.rand(1) * math.pi

angle_mult = torch.full((2, 10, 10), angle.item())
angle_mult[0] = torch.cos(angle_mult[0])
angle_mult[1] = torch.sin(angle_mult[1])
rot_mod_filter = cmplx_mult(mod_filter, angle_mult)

fig, axs = plt.subplots(2, 2)
print(mod_filter.size(), rot_mod_filter.size())
axs[0, 0].imshow(mod_filter[0, 0, 0])
axs[0, 1].imshow(mod_filter[1, 0, 0])
axs[1, 0].imshow(rot_mod_filter[0, 0, 0])
axs[1, 1].imshow(rot_mod_filter[1, 0, 0])
plt.title(f'angle={angle}')

plt.show()