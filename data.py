import argparse
import glob
import os
import PIL.Image as Image
import imageio
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from quicktorch.utils import imshow
import matplotlib.pyplot as plt


class TensorList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for i in range(len(self)):
            self[i] = self[i].to(device)
        return self


class CirrusDataset(Dataset):
    """Loads cirrus dataset from file.

    Args:
        img_dir (str): Path to dataset directory.
        transform (Trasform, optional): Transform(s) to
            be applied to the data.
        target_transform (Trasform, optional): Transform(s) to
            be applied to the targets.
    """
    def __init__(self, img_dir, indices=None, denoise=False, angle=False,
                 transform=None, target_transform=None):
        self.cirrus_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'input/*.png'))
        ]
        if denoise:
            self.mask_paths = [
                img for img in glob.glob(os.path.join(img_dir, 'clean/*.png'))
            ]
        else:
            self.mask_paths = [
                img for img in glob.glob(os.path.join(img_dir, 'target/*.png'))
            ]
        if angle:
            self.angles = torch.tensor(np.load(os.path.join(img_dir, 'angles.npy'))).unsqueeze(1)

        self.num_classes = 2
        self.transform = transform
        self.target_transform = target_transform
        self.angle = angle

        if indices is not None:
            self.cirrus_paths = [self.cirrus_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

    def __getitem__(self, i):
        cirrus = transforms.ToTensor()(
            Image.open(self.cirrus_paths[i])
        )
        mask = transforms.ToTensor()(
            Image.open(self.mask_paths[i])
        )
        if self.transform is not None:
            cirrus = self.transform(cirrus)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.angle:
            return cirrus, mask, self.angles[i]
        return cirrus, mask

    def __len__(self):
        return len(self.cirrus_paths)


class EMDataset(Dataset):
    """Loads ISBI EM dataset from file.

    Args:
        img_dir (str): Path to dataset directory.
        transform (Trasform, optional): Transform(s) to
            be applied to the data.
        target_transform (Trasform, optional): Transform(s) to
            be applied to the targets.
    """
    def __init__(self, img_dir,
                 transform=None, target_transform=None, aug_mult=4, indices=None):
        self.em_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'volume/*.png'))
        ]
        self.mask_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'labels/*.png'))
        ]

        self.test = False
        if len(self.mask_paths) == 0 and len(self.em_paths) > 0:
            self.test = True

        self.transform = transform
        self.aug_mult = aug_mult
        if indices is not None:
            self.em_paths = [self.em_paths[i] for i in indices]
            if not self.test:
                self.mask_paths = [self.mask_paths[i] for i in indices]

    def __getitem__(self, i):
        i = i // self.aug_mult
        em = np.array(Image.open(self.em_paths[i]))
        if self.test:
            mask = np.zeros_like(em)
        else:
            mask = np.array(Image.open(self.mask_paths[i]))

        if self.transform is not None:
            t = self.transform(image=em, mask=mask)
            em = t['image']
            mask = t['mask']
        em = np.expand_dims(em, axis=2)
        mask = np.expand_dims(mask, axis=2)
        return (
            transforms.ToTensor()(em),
            transforms.ToTensor()(mask)
        )

    def __len__(self):
        return len(self.em_paths) * self.aug_mult


def prepare_em_data(dir):
    """Rewrites tif files as pngs

    Args:
        dir (str): directory of data.
    """
    phase_types = [
        ('train', 'volume'),
        ('train', 'labels'),
        ('test', 'volume')
    ]

    for phase, type_img in phase_types:
        imgs = np.array(imageio.volread(os.path.join(dir, f'{phase}-{type_img}.tif')))

        if not os.path.isdir(os.path.join(dir, phase)):
            os.mkdir(os.path.join(dir, phase))
        if not os.path.isdir(os.path.join(dir, phase, type_img)):
            os.mkdir(os.path.join(dir, phase, type_img))

        for i, arr in enumerate(imgs):
            if phase == 'test':
                for j in range(4):
                    sub_arr = arr[256 * (j // 2):256 * (j // 2 + 1), 256 * (j % 2):256 * (j % 2 + 1)]
                    sub_img = Image.fromarray(sub_arr)
                    sub_img.save(os.path.join(dir, phase, type_img, f'{i*4 + j:05d}.png'))
            else:
                img = Image.fromarray(arr)
                img.save(os.path.join(dir, phase, type_img, f'{i:05d}.png'))


def post_em_data(dir):
    """Reads png files and saves as 30x512x512 tif file

    Args:
        dir (str): Directory of png files. Should be quarters of
            the image (topleft, topright, bottomleft, bottomright).
    """
    mask_paths = [
        img for img in glob.glob(os.path.join(dir, '*.png'))
    ]
    mask = None
    for i, path in enumerate(mask_paths):
        if i == 0:
            mask = np.expand_dims(np.array(Image.open(path)), 0)
        else:
            mask = np.concatenate((mask, np.expand_dims(np.array(Image.open(path)), 0)), axis=0)
    mask = np.reshape(mask, (30, 4, 256, 256))
    mask = np.concatenate((mask[:, :2], mask[:, 2:]), axis=2)
    mask = np.concatenate((mask[:, :1], mask[:, 1:]), axis=3)
    mask = mask.squeeze()
    imageio.volwrite(os.path.join(dir, 'out.tif'), mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a segmentation model')
    parser.add_argument('--prepare',
                        default=False, action='store_true',
                        help='Whether to prepare ISBI EM data. (default: %(default)s)')
    parser.add_argument('--dir',
                        default='data/isbi/test/labels', type=str,
                        help='Path to png images to convert to tif. (default: %(default)s)')
    parser.add_argument('--post',
                        default=False, action='store_true',
                        help='Whether to post ISBI EM data. (default: %(default)s)')
    parser.add_argument('--demo',
                        default=True, action='store_false',
                        help='Displays example from given dataset. (default: %(default)s)')
    parser.add_argument('--data',
                        default='cirrus', type=str,
                        choices=['cirrus', 'isbi'],
                        help='Type of dataset. Choices: %(choices)s (default: %(default)s)',
                        metavar='dataset')
    args = parser.parse_args()

    if args.prepare:
        prepare_em_data('data\isbi')
    elif args.post:
        post_em_data(args.dir)
    elif args.demo:
        if args.data == 'cirrus':
            data = DataLoader(CirrusDataset(
                    'data/halo1000/train'
                ),
                batch_size=1,
                shuffle=True
            )
        if args.data == 'isbi':
            data = DataLoader(EMDataset(
                    'data/isbi/train',
                    transform=transforms.RandomRotation(180),
                    target_transform=transforms.RandomRotation(180)
                ),
                batch_size=1,
                shuffle=True
            )

        example_img, example_mask = next(iter(data))
        imshow(example_img, example_mask)
