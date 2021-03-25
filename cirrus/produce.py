import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from igcn.seg.cmplxmodels import UNetIGCNCmplx
from data import CirrusDataset
from PIL import Image
from igcn.utils import _pair
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ast


def pad_for_crops(image, crop_size, overlap):
    image_size = image[0].shape
    leftover = ((crop_size - overlap) * (1 + image_size // (crop_size - overlap))) - image_size + overlap
    (pad_top, pad_left), (pad_bottom, pad_right) = leftover // 2, leftover - leftover // 2
    padded_image = [
        cv2.copyMakeBorder(image_channel, pad_top, pad_bottom, pad_left, pad_right, 4)
        for image_channel in image
    ]
    return np.array(padded_image)


def gen_crop_grid(size, crop_size, overlap):
    crop_grid = np.meshgrid(
        np.arange(0, size[0] - overlap[0], crop_size[0] - overlap[0]),
        np.arange(0, size[1] - overlap[1], crop_size[1] - overlap[1]),
    )
    return np.array(crop_grid).transpose()


def undo_padding(image, size, overlap):
    leftover = image.shape - size
    (pad_top, pad_left), (pad_bottom, pad_right) = leftover // 2, leftover - leftover // 2
    return image[pad_top:image.shape[0]-pad_bottom, pad_left:image.shape[1]-pad_right]


def dissect(image, size=(256, 256), overlap=None, downscale=1):
    """Split image into cropped sections.
    """
    size = np.array(_pair(size))
    crop_size = size * downscale
    if overlap is None:
        overlap = size[0] // 4
    overlap = np.array(_pair(overlap))
    image = pad_for_crops(image, crop_size, overlap)
    crop_grid = gen_crop_grid(image[0].shape, crop_size, overlap)
    batch_size = crop_grid.shape[0] * crop_grid.shape[1]
    batch = np.zeros((batch_size, image.shape[0], *size))
    crop_grid = crop_grid.reshape(-1, 2)
    for i in range(batch_size):
        cropped_image = image[:, crop_grid[i][0]:crop_grid[i][0]+crop_size[0], crop_grid[i][1]:crop_grid[i][1]+crop_size[1]]
        batch[i] = [cv2.resize(cropped_channel, tuple(size)) for cropped_channel in cropped_image]
    return torch.tensor(batch)


def stitch(batch, size, overlap=None, comb_fn=np.nanmean, downscale=1):
    """Stitches batch into an image of given size.
    """
    crop_size = batch.shape[-2] * downscale, batch.shape[-1] * downscale
    if overlap is None:
        overlap = crop_size[0] // 4
    size = np.array(size)
    overlap = np.array(_pair(overlap))
    # overlap = overlap // downscale
    # Image for each column strip
    out = np.full((1, *size), np.nan)
    print('before prepad', out.shape)
    out = pad_for_crops(out, crop_size, overlap)[0]
    print('after prepad', out.shape)
    crop_grid = gen_crop_grid(out.shape, crop_size, overlap)
    cols = np.full(
        (
            crop_grid.shape[0],
            crop_grid.shape[0] * (crop_size[0] - overlap[0]) + overlap[0],
            crop_size[0]
        ),
        np.nan
    )
    for i in range(crop_grid.shape[1]):
        for j in range(crop_grid.shape[0]):
            print(crop_grid[j][0][0], crop_grid[j][0][0]+crop_size[0])
            resized_img = cv2.resize(batch[j * crop_grid.shape[1] + i], crop_size)
            print(batch.shape, resized_img.shape)
            cols[j, crop_grid[j][0][0]:crop_grid[j][0][0]+crop_size[0], 0:crop_size[1]] = resized_img
        temp = out.copy()
        temp[:, crop_grid[0][i][1]:crop_grid[0][i][1]+crop_size[1]] = comb_fn(cols, axis=0)
        out = comb_fn(np.stack((out, temp)), axis=0)

    out = undo_padding(out, size, overlap)
    print('removed padding', out.shape)
    return out


def parse_filename(filename):
    def check_list(item):
        if item[0] == '[' and item[-1] == ']':
            return [i.strip() for i in ast.literal_eval(item)]
        return item
    ba_i = filename.index('bands') + len('bands') + 1
    ks_i = filename.index('kernel_size') + len('kernel_size') + 1
    ng_i = filename.index('no_g') + len('no_g') + 1
    bc_i = filename.index('base_channels') + len('base_channels') + 1
    ds_i = filename.index('downscale') + len('downscale') + 1
    return {
        'bands': check_list(filename[ba_i:filename.index('-', ba_i)]),
        'kernel_size': int(filename[ks_i:filename.index('-', ks_i)]),
        'no_g': int(filename[ng_i:filename.index('-', ng_i)]),
        'base_channels': int(filename[bc_i:filename.index('-', bc_i)]),
        'downscale': int(filename[ds_i:filename.index('_', ds_i)]),
    }


def pad(images, pad):
    padded_images = [
        [
            cv2.copyMakeBorder(image_channel, pad, pad, pad, pad, 4)
            for image_channel in image
        ]
        for image in images.numpy()
    ]
    padded_images = torch.tensor(padded_images)
    return padded_images


def get_model(path, n_classes=2, device='cpu'):
    name = os.path.split(path)[-1]
    params = parse_filename(name)
    n_channels = len(params['bands'])
    print(params)
    model = UNetIGCNCmplx(n_classes, n_channels, **params)
    model.load(save_path=path)
    model.to(device)
    params['name'] = name
    return model, params


def create_save_dir(name, save_dir='../data/cirrus_out'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))


def produce_output(
    model,
    dataset,
    downscale=1,
    n_channels=1,
    n_classes=1,
    padding=16,
    batch_size=1,
    device='cpu',
    galaxy='NGC3230'
):
    image, _ = dataset.get_galaxy(galaxy)
    print(image.size())

    original_shape = image[0].numpy().shape
    crop_size = (256, 256)
    padding = 16
    overlap = 128 * downscale
    batches = dissect(image.numpy(), size=crop_size, overlap=overlap, downscale=downscale)
    del(image)
    batches = batches.float()
    batches = batches.view(-1, batch_size, n_channels, *crop_size)
    batches_out = torch.zeros(batches.size(0), batch_size, n_classes, *crop_size)

    print(batches.size(), f'downscale={downscale}')
    # out1 = stitch(batches[:, 0].numpy(), original_shape, overlap=overlap, downscale=downscale)
    # out2 = stitch(batches[:, 1].numpy(), original_shape, overlap=overlap, downscale=downscale)

    # fig, axs = plt.subplots(2, 2)
    # axs[0][0].imshow(image[0])
    # axs[0][1].imshow(image[1])
    # axs[1][0].imshow(out1)
    # axs[1][1].imshow(out2)
    # plt.show()

    for i, batch in enumerate(batches):
        print(i)
        padded_batch = pad(batch, padding)
        print(padded_batch.min(), padded_batch.max(), padded_batch.mean())
        batch_out = model(padded_batch.to(device)).detach()
        print(batch_out.min(), batch_out.max(), batch_out.mean())
        batch_out = batch_out.squeeze(1)[..., padding:batch_out.size(-2)-padding, padding:batch_out.size(-1)-padding]
        batch_out = batch_out.cpu()
        batch_out = torch.clamp(batch_out, 0, 1)
        batch_out = torch.round(batch_out)
        batches_out[i] = batch_out
        # print(batches_out[i].shape)

        # batches_out[i] = padded_batch[0, 0, padding:padded_batch.size(-2)-padding, padding:padded_batch.size(-1)-padding]
        # print(batches_out[i].shape)

    del(batches)
    batches_out = batches_out.view(-1, n_classes, *crop_size).numpy()
    outs = []
    comb_fn = lambda x, axis: np.round(np.nanmean(x, axis))
    for i in range(n_classes):
        outs.append(stitch(batches_out[:, i], original_shape, overlap=overlap, downscale=downscale, comb_fn=comb_fn))
    del(batches_out)

    return outs


def save_outs(outs, labels, save_dir, name, galaxy):
    for t, label in zip(outs, labels):
        t = (t * 255).astype('uint8')
        t = Image.fromarray(t)
        if not os.path.isdir(os.path.join(save_dir, name)):
            os.makedirs(os.path.join(save_dir, name))
        t.save(os.path.join(save_dir, name, f'{galaxy}-{label}-{name}.png'))


def create_png_copies(dataset, save_dir, galaxy, labels):
    def norm(t):
        return (t - t.min()) / (t.max() - t.min())
    print(galaxy)
    image, target = dataset.get_galaxy(galaxy)
    print(image.size())
    bands = dataset.bands
    for i, band in enumerate(bands):
        t = image[i].numpy()
        t = (norm(t) * 255).astype('uint8')
        t = Image.fromarray(t)
        t.save(os.path.join(save_dir, 'copies', f'{galaxy}-{band}.png'))

    for i, label in enumerate(labels):
        t = target[i].numpy()
        t = (t * 255).astype('uint8')
        t = Image.fromarray(t)
        t.save(os.path.join(save_dir, 'copies', f'{galaxy}-annotation-{label}.png'))


def main():
    parser = argparse.ArgumentParser(description='Handles Cirrus segmentation tasks.')
    parser.add_argument('--survey_dir',
                        default='D:/MATLAS Data/FITS/matlas_reprocessed', type=str,
                        help='Path to survey directory. (default: %(default)s)')
    parser.add_argument('--mask_dir',
                        default='../data/cirrus', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--save_dir',
                        default='../data/cirrus_out', type=str,
                        help='Directory to save models to. (default: %(default)s)')
    parser.add_argument('--n_classes',
                        default=1, type=int,
                        help='Number of classes to predict. '
                             '(default: %(default)s)')
    parser.add_argument('--galaxy',
                        default='NGC3230', type=str,
                        help='Galaxy name.')
    parser.add_argument('--copies',
                        default=False, action='store_true',
                        help='Saves dataset input/target as png. (default: %(default)s)')
    args = parser.parse_args()

    path = r"C:\Users\Felix\Documents\igcn\models\seg\cirrus\cirrus_bands=['g', 'r']-kernel_size=3-no_g=4-base_channels=16-downscale=2_epoch120.pk"

    model, params = get_model(path, n_classes=args.n_classes, device='cuda:0')
    bands = params.pop('bands')
    downscale = params.pop('downscale')
    name = params.pop('name')
    labels = ('cirrus',)
    dataset = CirrusDataset(args.survey_dir, args.mask_dir, bands=bands)

    if args.copies:
        create_png_copies(dataset, args.save_dir, args.galaxy, labels)
        return

    outs = produce_output(
        model,
        dataset,
        downscale=downscale,
        n_channels=len(bands),
        n_classes=args.n_classes,
        device='cuda:0',
        galaxy=args.galaxy
    )
    save_outs(outs, labels, args.save_dir, name, args.galaxy)
    del(outs)


if __name__ == '__main__':
    main()