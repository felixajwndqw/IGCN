import os
import argparse
import torch
import torch.optim as optim
import albumentations
import PIL.Image as Image
from torch.utils.data import DataLoader
from igcn.seg.models import UNetIGCN
from quicktorch.utils import train, imshow
from data import EMDataset, post_em_data



def get_train_data():
    return DataLoader(
        EMDataset(
            'data/isbi/train',
            albumentations.Compose([
                albumentations.RandomCrop(256, 256),
                albumentations.Flip(),
                albumentations.RandomRotate90(),  # , fill=(0,)
                albumentations.ElasticTransform(),
                albumentations.PadIfNeeded(288, 288)
            ])
        ),
        batch_size=32,
        shuffle=True
    )


def get_test_data():
    return DataLoader(
        EMDataset(
            'data/isbi/test',
            albumentations.Compose([
                albumentations.PadIfNeeded(288, 288)
            ]),
            aug_mult=1,
        ),
        batch_size=8,
        # shuffle=True
    )


def produce_output(model=None, path=None, no_g=4, padding=16):
    if model is None and path is None:
        return TypeError('No model or path provided.')
    if model is not None and path is not None:
        return TypeError('Both model and path provided. Pass one.')
    if model is not None:
        name = model.name
    if path is not None:
        model = UNetIGCN(1, 1, no_g=no_g)
        model.load(save_path=path)
        name = os.path.split(path)[-1]
    imgs = get_test_data()

    save_dir = 'data/isbi/test/labels'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))

    for i, (batch, _) in enumerate(imgs):
        batch_out = model(batch)
        batch_out = batch_out.cpu().detach()
        batch_out = batch_out.squeeze(1)[..., padding:batch_out.size(-2)-padding, padding:batch_out.size(-1)-padding]
        batch_out = torch.clamp(batch_out, 0, 1)
        # batch_out = torch.round(batch_out)
        batch_out = batch_out.numpy()
        batch_out = (batch_out * 255).astype('uint8')
        for j, img in enumerate(batch_out):
            img = Image.fromarray(img)
            img.save(os.path.join(save_dir, name, f'{i*8 + j:05d}.png'))


def main():
    parser = argparse.ArgumentParser(description='Handles ISBI 2012 EM segmentation tasks.')
    parser.add_argument('--produce',
                        default=False, action='store_true',
                        help='Generates segmentations of test data and saves. (default: %(default)s)')
    parser.add_argument('--path',
                        default=None, type=str,
                        help='Path to trained model.')
    parser.add_argument('--full',
                        default=False, action='store_true',
                        help='Whether to run full test list. (default: %(default)s)')
    parser.add_argument('--no_g',
                        default=4, type=int,
                        help='Number of Gabor filters.')
    args = parser.parse_args()

    if args.produce:
        if args.path is None:
            raise TypeError('Empty --path argument.')
        produce_output(path=args.path)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_data = get_train_data()

        model = UNetIGCN(n_channels=1, n_classes=1, save_dir='models/seg/isbi', name='isbi',
            
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train(model, train_data, epochs=300, opt=optimizer, device=device, save_best=True)




if __name__ == '__main__':
    main()
