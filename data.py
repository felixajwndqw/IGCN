import argparse
import glob
import os
import albumentations
import PIL.Image as Image
import imageio
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from quicktorch.utils import imshow, train
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


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
                 transform=None, target_transform=None, aug_mult=4, indices=None, padding=0):
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

        self.padding = padding

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
        em = transforms.ToTensor()(em)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        return (em, mask)

    def __len__(self):
        return len(self.em_paths) * self.aug_mult


class PragueTextureDataset(Dataset):
    classes = {
        "bark": 0,
        "flowers": 1,
        "glass": 2,
        "man-made": 3,
        "nature": 4,
        "plants": 5,
        "rock": 6,
        "stone": 7,
        "textile": 8,
        "wood": 9
    }

    def __init__(self, data_dir, indices=None, padding=0, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.retrieve_paths(ET.parse(os.path.join(data_dir, 'data.xml')))
        self.transform = transform
        self.padding = padding
        if indices is not None:
            self.paths = [self.paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]
            self.class_maps = [self.class_maps[i] for i in indices]

    def __getitem__(self, i):
        image = np.array(Image.open(self.paths[i]))
        mask = np.array(Image.open(self.mask_paths[i]))

        n_regions = len(self.class_maps[i])
        regions = [np.where(mask == i, True, False) for i in range(n_regions)]

        for j in range(n_regions):
            mask[regions[j]] = self.classes[self.class_maps[i][j]]

        if self.transform is not None:
            t = self.transform(image=image, mask=mask)
            image = t['image']
            mask = t['mask']

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask).long()

        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)

        return image, mask

    def __len__(self):
        return len(self.paths)

    def retrieve_paths(self, metadata):
        sets = metadata.findall('section')
        self.paths = []
        self.mask_paths = []
        self.class_maps = []

        for i in range(len(sets) - 1):
            subsections = sets[i].findall('section')
            n_segments = int(subsections[0][1].text)
            for j in range(3):  # subsets - 3 masks
                for k in range(3):  # subsubsets - 3 imgs per mask
                    self.paths.append(os.path.join(self.data_dir, subsections[j * 5 + 2 + k][0].text.strip("\"")))
                    self.mask_paths.append(os.path.join(self.data_dir, subsections[j * 5 + 1][0].text.strip("\"")))
                    texture_sources = [subsections[j * 5 + 2 + k][l + 8].text for l in range(n_segments)]
                    self.class_maps.append([src.strip("\"").split('/')[0] for src in texture_sources])


def get_prague_splits(N, n_splits):
    """Removes test portion of every other sample before generating train/val
    """
    all_idxs = list(range(180))
    train_idxs = all_idxs[::2]
    test_idxs = all_idxs[1::2]

    kfold = KFold(n_splits=n_splits, shuffle=True)
    splits = kfold.split(train_idxs)
    splits = np.array(list(splits), dtype=object)

    return splits, test_idxs


def get_prague_train_data(args, training_args, data_dir, split):
    def grab_dataloader(isplit):
        return DataLoader(
            PragueTextureDataset(
                data_dir,
                transform=albumentations.Compose([
                    albumentations.Resize(size // downscale, size // downscale),
                    albumentations.Flip(),
                    albumentations.RandomRotate90(),
                    # albumentations.GaussNoise(p=1., var_limit=(0.05 * 255, 0.15 * 255)),
                    albumentations.PadIfNeeded(size // downscale + args.padding, size // downscale + args.padding, border_mode=4)
                ]),
                indices=isplit,
                padding=args.padding,
            ),
            batch_size=training_args.batch_size,
            shuffle=True
        )
    size = 512
    downscale = 4
    train_loader = grab_dataloader(split[0])
    val_loader = grab_dataloader(split[1])
    return train_loader, val_loader


def get_prague_test_data(args, training_args, data_dir, test_idxs=None):
    size = 512
    downscale = 4
    if test_idxs is None:
        test_idxs = np.arange(90, 180)
    return DataLoader(
        PragueTextureDataset(
            data_dir,
            transform=albumentations.Compose([
                albumentations.Resize(size // downscale, size // downscale),
                albumentations.PadIfNeeded(size // downscale + args.padding, size // downscale + args.padding, border_mode=4)
            ]),
            indices=test_idxs,
            padding=args.padding,
        ),
        batch_size=training_args.batch_size,
    )


def remove_padding(t, p):
    return t[..., p//2:-p//2, p//2:-p//2]


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
    print(mask_paths)
    mask = None
    for i, path in enumerate(mask_paths):
        if i == 0:
            mask = np.expand_dims(np.array(Image.open(path)), 0)
        else:
            mask = np.concatenate((mask, np.expand_dims(np.array(Image.open(path)), 0)), axis=0)
    print(mask.shape)
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
