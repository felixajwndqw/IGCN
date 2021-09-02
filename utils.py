import argparse
import math
import numpy as np
import PIL.Image as Image
import cv2
import random
import albumentations
import albumentations.augmentations.functional as F
from albumentations.core.transforms_interface import DualTransform, to_tuple


def parse_none(x):
    return None if x == 'None' else x


class ExperimentParser(argparse.ArgumentParser):
    """Constructs an argument parser that returns arguments in groups

    Arguments are divided into network and training arguments. This is just a
    LITTLE bit hacky :)
    """
    def __init__(self, description=''):
        super().__init__(description=description)
        self.n_parser = self.add_argument_group('net')
        self.t_parser = self.add_argument_group('training')
        self.construct_parsers()

    def construct_parsers(self):
        self.n_parser.add_argument(
            '--dataset',
            default='mnistrot', type=str,
            choices=['mnist', 'mnistrotated', 'mnistrot', 'mnistrp', 'cifar', 'isbi', 'bsd', 'cirrus', 'synth', 'prague'],
            help='Type of dataset. Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--model_type',
            default='SFC', type=str,
            choices=['SFC', 'IGCN', 'SFCNC', 'SFCResNet'],
            help='Model architecture to use. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--kernel_size',
            default=3, type=int,
            help='Kernel size')
        self.n_parser.add_argument(
            '--base_channels',
            default=16, type=int,
            help='Number of feature channels in first layer of network.')
        self.n_parser.add_argument(
            '--no_g',
            default=4, type=int,
            help='Number of Gabor filters.')
        self.n_parser.add_argument(
            '--dropout',
            default=0., type=float,
            help='Probability of dropout layer(s).')
        self.n_parser.add_argument(
            '--inter_gp',
            default=None, type=parse_none,
            choices=[None, 'max', 'mag', 'avg', 'sum'],
            help='Type of pooling to apply across Gabor '
                 'axis for intermediate layers. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--final_gp',
            default=None, type=parse_none,
            choices=[None, 'max', 'mag', 'avg', 'sum'],
            help='Type of pooling to apply across Gabor axis for the final layer. '
                 '(default: %(default)s)')
        self.n_parser.add_argument(
            '--all_gp',
            default=False, action='store_true',
            help='Whether to apply Gabor pooling on all layers')
        self.n_parser.add_argument(
            '--relu_type',
            default='c', type=str,
            choices=['c', 'mod', 'z', 'mf'],
            help='Type of relu layer. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--fc_type',
            default='cat', type=str,
            choices=['cat', 'mag'],
            help='How complex tensors are combined into real tensors prior to FC. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--fc_block',
            default='lin', type=str,
            choices=['lin', 'clin', 'cnv', 'bmp', 'cmp', 'nmp'],
            help='Linear, complex linear or 1x1 conv for FC layers. Former will not be applied '
                 'before projection.'
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--fc_relu_type',
            default='c', type=str,
            choices=['c', 'mod', 'z', 'mf'],
            help='Type of relu layer for FC layers.'
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--mod',
            default='hadam', type=str,
            choices=['hadam', 'cmplx'],
            help='How to apply modulation to filters (hadamard or complex multiplication).'
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--cmplx',
            default=True, action='store_false',
            help='Whether to use a complex architecture.')
        self.n_parser.add_argument(
            '--single',
            default=False, action='store_true',
            help='Whether to use a single gconv layer between each pooling layer.')
        self.n_parser.add_argument(
            '--pooling',
            default='max', type=str,
            choices=['max', 'mag', 'avg'],
            help='Type of pooling. Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--nfc',
            default=2, type=int,
            help='Number of fully connected layers before classification.')
        self.n_parser.add_argument(
            '--bnorm',
            default='new', type=str,
            choices=['new', 'old'],
            help='New = proper bnorm /w momentum. Old = plain normalisation.')
        self.n_parser.add_argument(
            '--weight_init',
            default='he',
            type=parse_none,
            choices=[None, 'he', 'glorot'],
            help=('Type of weight initialisation. Choices: %(choices)s '
                  '(default: %(default)s, corresponding to re/im independent He init.)'))
        self.n_parser.add_argument(
            '--softmax',
            default=False, action='store_true',
            help='Whether to use softmax for final classification.')
        self.n_parser.add_argument(
            '--group',
            default=False, action='store_true',
            help='Whether to use group convolutions.')
        self.n_parser.add_argument(
            '--upsample_mode',
            default='bilinear',
            type=parse_none,
            choices=[None, 'nearest', 'bilinear'],
            help=('Decoder upsampling method. Choices: %(choices)s '
                  '(default: %(default)s.)'))
        self.n_parser.add_argument(
            '--l_init',
            default='uniform', type=str,
            choices=['uniform', 'normal', 'normalg', 'fixed'],
            help='Initialisation method for gabor lambda param.')
        self.n_parser.add_argument(
            '--sigma_init',
            default='fixed', type=str,
            choices=['fixed', 'normal'],
            help='Initialisation method for gabor lambda param.')
        self.n_parser.add_argument(
            '--single_param',
            default=False, action='store_true',
            help='Uses only a single lambda/sigma parameter.')

        self.t_parser.add_argument(
            '--epochs',
            default=250, type=int,
            help='Number of epochs to train over.')
        self.t_parser.add_argument(
            '--augment',
            default=False, action='store_true',
            help='Whether to apply data augmentation.')
        self.t_parser.add_argument(
            '--lr',
            default=1e-4, type=float,
            help='Learning rate.')
        self.t_parser.add_argument(
            '--lr_decay',
            default=1, type=float,
            help='Learning rate decay.')
        self.t_parser.add_argument(
            '--weight_decay',
            default=1e-7, type=float,
            help='Weight decay/l2 reg.')
        self.t_parser.add_argument(
            '--nsplits',
            default=1, type=int,
            help='Number of validation splits to train over')
        self.t_parser.add_argument(
            '--batch_size',
            default=32, type=int,
            help='Number of samples in each batch')
        self.t_parser.add_argument(
            '--translate',
            default=0, type=float,
            help='Translation coefficient for data augmentation.')
        self.t_parser.add_argument(
            '--scale',
            default=0, type=float,
            help='Scale coefficient for data augmentation.')
        self.t_parser.add_argument(
            '--shear',
            default=0, type=float,
            help='Shear coefficient for data augmentation.')
        self.t_parser.add_argument(
            '--name',
            default=None,
            type=parse_none,
            help='Name to save model under.')
        self.t_parser.add_argument(
            '--debug',
            default=False, action='store_true',
            help='Runs a single split with no metric writing.')
        self.t_parser.add_argument(
            '--eval_best',
            default=False, action='store_true',
            help='Evaluates only model from best split/.')

    def parse_group_args(self):
        args = self.parse_args()
        arg_groups = {}

        for group in self._action_groups:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            arg_groups[group.title] = argparse.Namespace(**group_dict)

        return arg_groups['net'], arg_groups['training']

    @staticmethod
    def args_to_str(args):
        kwargs = vars(args)
        return '_'.join([f'{key}={val}' for key, val in kwargs.items()])


class AlbumentationsWrapper():
    """Wraps albumentations transforms for use with torchvision
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        out = self.transform(image=np.array(image))
        return Image.fromarray(out['image'])


class RotateAndCrop(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.
    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height,
        width,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.,
    ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        img = F.rotate(img, angle, interpolation, self.border_mode, self.value)
        img = self._remove_rotation_padding(img, angle)
        img = F.random_crop(img, self.height, self.width, h_start, w_start)
        return img

    def apply_to_mask(self, img, angle=0, h_start=0, w_start=0, **params):
        img = F.rotate(img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        img = self._remove_rotation_padding(img, angle)
        return F.random_crop(img, self.height, self.width, h_start, w_start)

    def _remove_rotation_padding(self, img, angle):
        h, w = img.shape[0], img.shape[1]
        crop_w, crop_h = _largest_rotated_rect(w, h, np.deg2rad(angle))
        return img[
            (h - crop_h) // 2:(h - crop_h) // 2 + crop_h,
            (w - crop_w) // 2:(w - crop_w) // 2 + crop_w
        ]

    def get_params(self):
        return {
            "angle": random.choice([i * 180 / 8 - 180 for i in range(16)]),
            "h_start": random.random(),
            "w_start": random.random()
        }

    def apply_to_bbox(self, bbox, angle=0, **params):
        return F.bbox_rotate(bbox, angle, params["rows"], params["cols"])

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return F.keypoint_rotate(keypoint, angle, **params)

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")


def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        int(bb_w - 2 * x),
        int(bb_h - 2 * y)
    )


def calculate_error(items):
    N = len(items)
    mean_items = sum(items) / N
    diff_sq_sum = sum((item - mean_items) ** 2 for item in items)
    return math.sqrt(diff_sq_sum / (N * (N - 1)))


def main():
    parser = ExperimentParser()
    net_args, training_args = parser.parse_group_args()
    print("Net args")
    print(parser.args_to_str(net_args))
    print("Training args")
    print(parser.args_to_str(training_args))


if __name__ == '__main__':
    main()
