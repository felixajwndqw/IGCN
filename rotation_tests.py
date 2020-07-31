import os
import torch
import PIL
import numpy as np
from utils import ExperimentParser
from quicktorch.data import mnist
from igcn.models import IGCN
import matplotlib.pyplot as plt
import torchgeometry as tgm
from igcn.cmplx import magnitude, cmplx, phase
from sklearn.manifold import TSNE


MODEL = 'models/mnistrp/dataset=mnistrp_kernel_size=3_base_channels=96_no_g=8_dropout=0.0_inter_gp=max_final_gp=max_all_gp=False_relu_type=mod_fc_type=cat_fc_block=lin_cmplx=True_single=False_pooling=avg_nfc=3_bnorm=new_weight_init=he_epoch300.pk'
mags = []
tsnes = 0


def get_image(n_examples=50, numbers=range(10), collapse=False):
    _, loader, _ = mnist(batch_size=2000)
    imgs, lbls = next(iter(loader))
    out_imgs = []
    # retrieve one example of each number
    for number in numbers:
        count = 0
        out_imgs.append([])
        for img, lbl in zip(imgs, lbls):
            if lbl[number]:
                out_imgs[-1].append(img)
                count += 1
                if count == n_examples:
                    out_imgs[-1] = torch.stack(out_imgs[-1])
                    break
    if collapse:
        return torch.cat(out_imgs)
    return torch.stack(out_imgs)


def rotate(img, theta):
    if not isinstance(theta, torch.Tensor):
        if np.isscalar(theta):
            theta = [theta]
    if img.dim() == 3:
        img = img.unsqueeze(0)
    rot = torch.empty_like(img)
    _, _, rows, cols = img.size()
    centre = rot.new_tensor(((rows - 1) / 2, (cols - 1) / 2)).unsqueeze(0).repeat(len(theta), 1)
    scale = rot.new_ones(len(theta))
    theta = rot.new_tensor(theta)
    M = tgm.get_rotation_matrix2d(centre, theta, scale)
    if img.size(0) != M.size(0):
        M = M.repeat(img.size(0), 1, 1)
    rot = tgm.warp_affine(img, M, dsize=(rows, cols))
    return rot


def get_model(model_path=MODEL):
    model = IGCN(
        dataset='mnistrp',
        kernel_size=3,
        base_channels=96,
        no_g=8,
        dropout=0.0,
        inter_gp='max',
        final_gp='max',
        all_gp=False,
        relu_type='mod',
        fc_type='cat',
        fc_block='lin',
        cmplx=True,
        single=False,
        pooling='avg',
        nfc=3,
        weight_init='he',
    )
    model.load(save_path=model_path)
    return model


def total_magnitude(self, input, output):
    global mags
    in_mag = magnitude(input[0]).cpu().detach().mean().numpy()
    out_mag = magnitude(output).cpu().detach().mean().numpy()
    mags[-1].append(out_mag / in_mag)


def tsne(self, input, output):
    global tsne_xs, tsne_mags, tsne_phases
    X = output.cpu().detach().view(output.size(1), -1)
    mags = magnitude(output).cpu().detach().view(output.size(1), -1)
    phases = phase(output).cpu().detach().view(output.size(1), -1)
    tsne_xs = TSNE(n_components=1).fit_transform(X)
    tsne_mags = TSNE(n_components=1).fit_transform(mags)
    tsne_phases = TSNE(n_components=1).fit_transform(phases)


def main():
    global tsne_xs, tsne_mags, tsne_phases
    parser = ExperimentParser(description='Runs rotation tests')
    parser.add_argument('--image',
                        default='', type=str,
                        help='Path to image. (default: %(default)s)')
    parser.add_argument('--model_path',
                        default='', type=str,
                        help='Path to image. (default: %(default)s)')
    parser.add_argument('--test',
                        default='rm', choices=['rm', 'tsne'], type=str,
                        help='Type of test to run. (default: %(default)s)')

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()

    model = get_model()

    if args.test == 'tsne':
        n_examples = 50
        numbers = [6, 9]
        # numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        imgs = get_image(n_examples=n_examples, numbers=numbers, collapse=True)
        model.conv1.register_forward_hook(tsne)
        thetas = torch.randn(imgs.size(0)) * 360 - 180
        r_imgs = rotate(imgs, thetas)
        out = model(r_imgs)

        tsne_xs = (tsne_xs - tsne_xs.min()) / (tsne_xs.max() - tsne_xs.min())
        tsne_mags = (tsne_mags - tsne_mags.min()) / (tsne_mags.max() - tsne_mags.min())
        tsne_phases = (tsne_phases - tsne_phases.min()) / (tsne_phases.max() - tsne_phases.min())

        fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
        for axi, tsnes in zip(ax, (tsne_xs, tsne_mags, tsne_phases)):
            for i in range(len(numbers)):
                axi.plot(thetas[i*n_examples:(i+1)*n_examples].numpy(), tsnes[i*n_examples:(i+1)*n_examples], 'o')
        plt.tight_layout()
        plt.show()


    if args.test == 'rm':
        imgs = get_image()
        model.conv1.register_forward_hook(total_magnitude)
        thetas = np.linspace(0, 360, 720)

        for i, img in enumerate(imgs):
            print(f'Starting image #{i}')
            mags.append([])
            for t in thetas:
                r_img = rotate(img, t)
                out = model(r_img)

        fig = plt.figure()
        for i, mags_i in enumerate(mags):
            plt.plot(thetas, mags_i, label=str(i))
        plt.legend()
        plt.xticks(np.linspace(0, 360, 9))
        plt.ylim(np.min(mags) - .1, np.max(mags) + .1)
        plt.ylabel('Response Magnitude')
        plt.xlabel('Rotation Angle')
        plt.tight_layout()
        plt.savefig('figs/responsemagnitude.png')

        fig2 = plt.figure()
        plt.boxplot(mags)
        plt.ylabel('Response Magnitude')
        plt.xlabel('Number')
        plt.xticks(range(1, 11), range(0, 10))
        plt.tight_layout()
        plt.savefig('figs/responsemagnitudespreads.png')
        plt.show()




    # print(out.mean(), out.max(), out.min())
    # plt.imshow(out.detach().cpu().numpy())
    # plt.show()



if __name__ == '__main__':
    main()