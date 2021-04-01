import ast
import glob
import os

import numpy as np
import PIL.Image as Image
import torch

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import warnings

warnings.simplefilter('ignore', category=AstropyWarning)


class TensorList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for i in range(len(self)):
            self[i] = self[i].to(device)
        return self


class CirrusDataset(Dataset):
    """
    Dataset class for Cirrus data.

    Args:
        survey_dir (str): Path to survey directory.
        mask_dir (str): Path to mask directory.
        indices (array-like, optional): Indices of total dataset to use.
            Defaults to None.
        num_classes (int, optional): Number of classes. Defaults to 2.
        transform (Trasform, optional): Transform(s) to
            be applied to the data. Defaults to None.
        target_transform (Trasform, optional): Transform(s) to
            be applied to the targets. Defaults to None.
        crop (float, optional): Degrees to crop around centre. Defaults to .5.
    """

    means = {
        # processed cirrus+HB
        # 'g': .354,
        # 'r': .404,
        # processed cirrus
        # 'g': .254,
        # 'r': .301,
        # 'i': .246,
        # 'gr': .276,
        # reprocessed cirrus
        'g': 0.288,
        'r': 0.207,
    }
    stds = {
        # processed cirrus+HB
        # 'g': .759,
        # 'r': .924,
        # processed cirrus
        # 'g': .741,
        # 'r': .903,
        # 'i': 1.063,
        # 'gr': .744,
        # reprocessed cirrus
        'g': 0.712,
        'r': 0.795,
    }

    def __init__(self, survey_dir, mask_dir, indices=None, num_classes=1,
                 transform=None, target_transform=None, crop_deg=.5,
                 aug_mult=2, bands='g'):
        self.all_mask_paths = [
            array for array in glob.glob(os.path.join(mask_dir, '*.npz'))
        ]

        if type(bands) is str:
            bands = [bands]

        self.galaxies = []
        self.cirrus_paths = []
        self.mask_paths = []
        for i, mask_path in enumerate(self.all_mask_paths):
            mask_args = self.decode_filename(mask_path)
            galaxy = mask_args['name']
            fits_dirs = [os.path.join(
                survey_dir,
                galaxy,
                band
            ) for band in bands]
            if all(os.path.isdir(path) for path in fits_dirs):
                fits_paths = [glob.glob(path + '/*.fits')[0] for path in fits_dirs]
                self.galaxies.append(galaxy)
                self.cirrus_paths.append(fits_paths)
                self.mask_paths.append(mask_path)

        self.num_classes = num_classes
        self.bands = bands
        self.num_channels = len(bands)
        self.transform = transform
        self.norm_transform = transforms.Normalize(
            tuple(self.means[b] for b in self.bands),
            tuple(self.stds[b] for b in self.bands)
        )
        self.crop_deg = crop_deg
        self.aug_mult = aug_mult

        if indices is not None:
            self.cirrus_paths = [self.cirrus_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

    def __getitem__(self, i):
        i = i // self.aug_mult
        cirrus = [fits.open(path)[0] for path in self.cirrus_paths[i]]
        wcs = WCS(cirrus[0].header, naxis=2)
        mask, centre = self.decode_np_mask(np.load(self.mask_paths[i]))
        # if not (mask.shape[0] == self.num_classes or mask.shape[-1] == self.num_classes):
        #     raise ValueError(f'Mask {mask.shape} does not match number of channels ({self.num_classes})')

        mask = mask[:self.num_classes]

        if self.crop_deg is not None:
            cirrus = np.array([self.crop(ci.data, wcs, centre) for ci in cirrus])
            mask = self.crop(mask, wcs, centre)
        else:
            cirrus = np.array([ci.data for ci in cirrus])

        # cirrus = cirrus.reshape(cirrus.shape[-2], cirrus.shape[-1], -1)
        cirrus = cirrus.transpose((1, 2, 0))
        cirrus = cirrus.astype('float32')
        # mask = mask.reshape(mask.shape[-2], mask.shape[-1], -1)
        mask = mask.transpose((1, 2, 0))
        mask = mask.astype('float32')

        if self.transform is not None:
            t = self.transform(image=cirrus, mask=mask)
            cirrus = t['image']
            mask = t['mask']
        return (
            # self.norm_transform(transforms.ToTensor()(cirrus)),
            transforms.ToTensor()(cirrus),
            transforms.ToTensor()(mask)
        )

    def __len__(self):
        return len(self.cirrus_paths) * self.aug_mult

    def crop(self, image, wcs, centre):
        def fit_image(bbox, image_shape):
            bbox[:, 1] = np.clip(bbox[:, 1], 0, image_shape[-2])
            bbox[:, 0] = np.clip(bbox[:, 0], 0, image_shape[-1])
            return bbox

        bbox = wcs.wcs_world2pix(
            [
                centre - self.crop_deg / 2,
                centre + self.crop_deg / 2
            ],
            0
        ).astype(np.int32)
        bbox = fit_image(bbox, image.shape)
        return image[..., bbox[0, 1]:bbox[1, 1], bbox[1, 0]:bbox[0, 0]].copy()

    def get_galaxy(self, galaxy):
        try:
            index = self.galaxies.index(galaxy)
        except ValueError:
            print(f'Galaxy {galaxy} not stored in this dataset.')
            return None
        return self[index * self.aug_mult]

    @classmethod
    def decode_filename(cls, path):
        def check_list(item):
            if item[0] == '[' and item[-1] == ']':
                return [i.strip() for i in ast.literal_eval(item)]
            return item
        filename = os.path.split(path)[-1]
        filename = filename[:filename.rfind('.')]
        pairs = [pair.split("=") for pair in filename.split("-")]
        args = {key: check_list(val) for key, val in pairs}
        return args

    @classmethod
    def decode_np_mask(cls, array):
        shape, mask, centre = array['shape'], array['mask'], array['centre']
        mask = np.unpackbits(mask)
        mask = mask[:np.prod(shape)]
        return mask.reshape(shape), centre


class SynthCirrusDataset(Dataset):
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
                img for img in glob.glob(os.path.join(img_dir, 'cirrus_mask/*.png'))
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
        cirrus = np.array(Image.open(self.cirrus_paths[i]))
        mask = np.array(Image.open(self.mask_paths[i]))
        if self.transform is not None:
            t = self.transform(image=cirrus, mask=mask)
            cirrus = t['image']
            mask = t['mask']
        cirrus = transforms.ToTensor()(cirrus)
        mask = transforms.ToTensor()(mask)
        if self.angle:
            return cirrus, mask, self.angles[i]
        return cirrus, mask

    def __len__(self):
        return len(self.cirrus_paths)