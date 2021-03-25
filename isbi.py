import argparse
import os
import math
import random
import time
import torch
import torch.optim as optim
import albumentations
import PIL.Image as Image
from torch.utils.data import DataLoader
from igcn.seg.models import UNetIGCN
from igcn.seg.cmplxmodels import UNetIGCNCmplx
from quicktorch.utils import train, imshow
from data import EMDataset, post_em_data


def get_train_data(batch_size=8):
    train_idxs = list(range(30))
    valid_idxs = [train_idxs.pop(random.randint(0, 29))]
    print(valid_idxs)
    trainloader = DataLoader(
        EMDataset(
            'data/isbi/train',
            albumentations.Compose([
                albumentations.RandomCrop(256, 256),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.ElasticTransform(),
                albumentations.PadIfNeeded(288, 288, border_mode=4)
            ]),
            indices=train_idxs
        ),
        batch_size=batch_size,
        shuffle=True
    )
    validloader = DataLoader(
        EMDataset(
            'data/isbi/train',
            albumentations.Compose([
                albumentations.RandomCrop(256, 256),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.ElasticTransform(),
                albumentations.PadIfNeeded(288, 288, border_mode=4)
            ]),
            indices=valid_idxs
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return trainloader, validloader


def get_test_data(batch_size=8):
    return DataLoader(
        EMDataset(
            'data/isbi/test',
            albumentations.Compose([
                albumentations.PadIfNeeded(288, 288, border_mode=4)
            ]),
            aug_mult=1,
        ),
        batch_size=batch_size
    )


def parse_filename(filename):
    ks_i = filename.index('kernel_size') + len('kernel_size') + 1
    ng_i = filename.index('no_g') + len('no_g') + 1
    bc_i = filename.index('base_channels') + len('base_channels') + 1
    return {
        'kernel_size': int(filename[ks_i:filename.index('_', ks_i)]),
        'no_g': int(filename[ng_i:filename.index('_', ng_i)]),
        'base_channels': int(filename[bc_i:filename.index('_', bc_i)]),
    }


def produce_output(model=None, path=None, padding=16, batch_size=8, device='cpu'):
    if model is None and path is None:
        return TypeError('No model or path provided.')
    if model is not None and path is not None:
        return TypeError('Both model and path provided. Pass one.')
    if model is not None:
        name = model.name
    if path is not None:
        name = os.path.split(path)[-1]
        params = parse_filename(name)
        print(params)
        model = UNetIGCNCmplx(1, 1, **params)
        model.load(save_path=path)
    imgs = get_test_data(batch_size)
    model.to(device)

    save_dir = 'data/isbi/test/labels'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))

    for i, (batch, _) in enumerate(imgs):
        batch_out = model(batch.to(device))
        batch_out = batch_out.cpu().detach()
        batch_out = batch_out.squeeze(1)[..., padding:batch_out.size(-2)-padding, padding:batch_out.size(-1)-padding]
        batch_out = torch.clamp(batch_out, 0, 1)
        batch_out = batch_out.numpy()
        batch_out = (batch_out * 255).astype('uint8')
        for j, img in enumerate(batch_out):
            img = Image.fromarray(img)
            img.save(os.path.join(save_dir, name, f'{i*batch_size + j:05d}.png'))


def write_results(dset, kernel_size, no_g, base_channels, m, no_epochs,
                  total_params, mins, secs, cmplx=False,
                  best_split=1, nsplits=1, error_m=None):
    f = open("seg_results.txt", "a+")
    out = (
        "\n" + dset +
        "," + str(kernel_size) +
        "," + str(no_g) +
        "," + str(base_channels) +
        "," + str(cmplx) +
        ',' + "{:1.4f}".format(m['accuracy']) +
        "," + str(m['epoch']) +
        "," + str(no_epochs) +
        "," + str(best_split) +
        "," + str(nsplits) +
        ',' + "{:1.4f}".format(total_params) +
        ',' + "{:3d}m{:2d}s".format(mins, secs)
    )
    if error_m is not None:
        out += ',' + "{:1.4f}".format(error_m['accuracy'])
    f.write(out)
    f.close()


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
    parser.add_argument('--kernel_size',
                        default=3, type=int,
                        help='Kernel size of filters.')
    parser.add_argument('--base_channels',
                        default=16, type=int,
                        help='Number of filter channels to start network with. e.g. 16 becomes 16>32>48>64>64>64>48>32>16.')
    parser.add_argument('--cmplx',
                        default=False, action='store_true',
                        help='Whether to use a complex architecture.')

    parser.add_argument('--epochs',
                        default=250, type=int,
                        help='Number of epochs to train over.')
    parser.add_argument('--batch_size',
                        default=8, type=int,
                        help='Number of samples in each batch.')
    parser.add_argument('--splits',
                        default=1, type=int,
                        help='Number of validation splits to train over')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.produce:
        if args.path is None:
            raise TypeError('Empty --path argument.')
        produce_output(path=args.path, batch_size=args.batch_size, device=device)
    else:
        metrics = []
        for i in range(args.splits):
            train_data, valid_data = get_train_data(batch_size=args.batch_size)

            if args.cmplx:
                Net = UNetIGCNCmplx
            else:
                Net = UNetIGCN
            model = Net(
                n_channels=1,
                n_classes=1,
                save_dir='models/seg/isbi',
                name=f'isbi_kernel_size={args.kernel_size}_no_g={args.no_g}_base_channels={args.base_channels}',
                no_g=args.no_g,
                kernel_size=args.kernel_size,
                base_channels=args.base_channels
            ).to(device)

            total_params = sum(p.numel()
                               for p in model.parameters()
                               if p.requires_grad) / 1000000
            print("Total # parameter: " + str(total_params) + "M")

            optimizer = optim.Adam(model.parameters(), lr=1e-2)

            start = time.time()
            m = train(
                model,
                [train_data, valid_data],
                epochs=args.epochs,
                opt=optimizer,
                device=device,
                save_best=True
            )

            time_taken = time.time() - start
            mins = int(time_taken // 60)
            secs = int(time_taken % 60)
            del(model)
            torch.cuda.empty_cache()
            metrics.append(m)

            if args.produce:
                model.name += f'_epoch{m["epoch"]}.pk'
                produce_output(model, device=device)
                post_em_data(os.path.join('data/isbi/test/labels', model.name))

        mean_m = {key: sum(mi[key] for mi in metrics) / args.splits for key in m.keys()}
        best_acc = max([mi['accuracy'] for mi in metrics])
        best_split = [mi['accuracy'] for mi in metrics].index(best_acc) + 1
        mean_m['epoch'] = metrics[best_split-1]['epoch']
        error_m = None
        if args.splits > 1:
            error_m = {key: math.sqrt(sum((mi[key] - mean_m[key]) ** 2 for mi in metrics) / (args.splits * (args.splits - 1)))
                       for key in m.keys()}
        write_results(
            'isbi',
            args.kernel_size,
            args.no_g,
            args.base_channels,
            mean_m,
            args.epochs,
            total_params,
            mins,
            secs,
            cmplx=args.cmplx,
            nsplits=args.splits,
            best_split=best_split,
            error_m=error_m
        )



if __name__ == '__main__':
    main()
