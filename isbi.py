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
from igcn.seg.models import UNetIGCN, UNetIGCNCmplx
from quicktorch.utils import train, imshow
from data import EMDataset, post_em_data
from utils import ExperimentParser


def get_isbi_train_data(args, training_args, data_dir='../data/isbi', split=None, **kwargs):
    if split is not None:
        if split[0] is None:
            split[0] = list(range(30))
            split[1] = [split[0].pop(random.randint(0, len(split[0]) - 1)) for _ in range(args.val_size)]
    trainloader = DataLoader(
        EMDataset(
            data_dir + '/train',
            albumentations.Compose([
                albumentations.RandomCrop(256, 256),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.ElasticTransform(alpha=2),
                albumentations.GaussNoise(p=1., var_limit=(0.05 * 255, 0.15 * 255)),
                albumentations.PadIfNeeded(256 + args.padding, 256 + args.padding, border_mode=4)
            ]),
            indices=split[0],
            padding=padding
        ),
        batch_size=training_args.batch_size,
        shuffle=True
    )
    validloader = DataLoader(
        EMDataset(
            data_dir + '/train',
            albumentations.Compose([
                albumentations.RandomCrop(256, 256),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.ElasticTransform(alpha=2),
                albumentations.GaussNoise(p=1., var_limit=(0.05 * 255, 0.15 * 255)),
                albumentations.PadIfNeeded(256 + args.padding, 256 + args.padding, border_mode=4)
            ]),
            indices=split[1],
            padding=padding
        ),
        batch_size=training_args.batch_size,
        shuffle=True
    )
    return trainloader, validloader


def get_isbi_test_data(args, training_args, data_dir='../data/isbi', **kwargs):
    padding = 64
    return DataLoader(
        EMDataset(
            data_dir + '/test',
            albumentations.Compose([
                albumentations.PadIfNeeded(256 + padding, 256 + padding, border_mode=4)
            ]),
            aug_mult=1,
            padding=padding
        ),
        batch_size=training_args.batch_size
    )


# def parse_filename(filename):
#     ks_i = filename.index('kernel_size') + len('kernel_size') + 1
#     ng_i = filename.index('no_g') + len('no_g') + 1
#     bc_i = filename.index('base_channels') + len('base_channels') + 1
#     return {
#         'kernel_size': int(filename[ks_i:filename.index('_', ks_i)]),
#         'no_g': int(filename[ng_i:filename.index('_', ng_i)]),
#         'base_channels': int(filename[bc_i:filename.index('_', bc_i)]),
#     }



def parse_filename(filename):
    def check_list(item):
        if item[0] == '[' and item[-1] == ']':
            return [i.strip() for i in ast.literal_eval(item)]
        return item
    # ba_i = filename.index('bands') + len('bands') + 1
    ks_i = filename.index('kernel_size') + len('kernel_size') + 1
    ng_i = filename.index('no_g') + len('no_g') + 1
    bc_i = filename.index('base_channels') + len('base_channels') + 1
    # ds_i = filename.index('downscale') + len('downscale') + 1
    params = {
        'variant': filename.split('-')[0],
        # 'bands': check_list(filename[ba_i:filename.index('-', ba_i)]),
        'kernel_size': int(filename[ks_i:filename.index('-', ks_i)]),
        'no_g': int(filename[ng_i:filename.index('-', ng_i)]),
        'base_channels': int(filename[bc_i:filename.index('-', bc_i)]),
    }
    # if 'gp' in filename or 'relu' in filename:
    #     params['downscale'] = int(filename[ds_i:filename.index('-', ds_i)])
    # else:
    #     params['downscale'] = int(filename[ds_i:filename.index('_', ds_i)])
    if 'gp' in filename:
        gp_i = filename.index('gp') + len('gp') + 1
        params['final_gp'] = filename[gp_i:filename.index('-', gp_i)]
    if 'relu' in filename:
        relu_i = filename.index('relu') + len('relu') + 1
        params['relu_type'] = filename[relu_i:filename.index('_', relu_i)]

    params['name'] = os.path.split(filename)[-1]

    return params


def produce_output(args, training_args, net_args, model=None, path=None, padding=32, batch_size=8, device='cpu'):
    if model is None and path is None:
        return TypeError('No model or path provided.')
    if model is not None and path is not None:
        return TypeError('Both model and path provided. Pass one.')
    if model is not None:
        name = model.name
    if path is not None:
        name = os.path.split(path)[-1]
        params = parse_filename(name)
        params['pooling'] = net_args.pooling
        print(params)
        model = UNetIGCNCmplx(1, 1, **params)
        model.load(save_path=path)
    imgs = get_isbi_test_data(args, training_args)
    model.to(device)

    save_dir = 'data/isbi/test/labels'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    if not os.path.isdir(os.path.join(save_dir, name)):
        os.makedirs(os.path.join(save_dir, name))

    for i, (batch, _) in enumerate(imgs):
        batch_out = model(batch.to(device))
        batch_out = batch_out.cpu().detach()
        batch_out = batch_out.squeeze(1)
        if padding > 0:
            batch_out = batch_out[..., padding:batch_out.size(-2)-padding, padding:batch_out.size(-1)-padding]
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
    parser = ExperimentParser(description='Handles ISBI 2012 EM segmentation tasks.')
    parser.add_argument('--produce',
                        default=False, action='store_true',
                        help='Generates segmentations of test data and saves. (default: %(default)s)')
    parser.add_argument('--path',
                        default=None, type=str,
                        help='Path to trained model.')
    parser.add_argument('--val_size',
                        default=1, type=int,
                        help='Number of images used for validation dataset.')
    parser.add_argument('--padding',
                        default=32, type=int,
                        help='Number of images used for validation dataset.')

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.produce:
        if args.path is None:
            raise TypeError('Empty --path argument.')
        produce_output(args, training_args, net_args, path=args.path, batch_size=args.batch_size, device=device, padding=args.padding)
    else:
        metrics = []
        for i in range(args.splits):
            train_idxs = list(range(30))
            valid_idxs = [train_idxs.pop(random.randint(0, 29)) for _ in range(args.val_size)]
            train_data, valid_data = get_isbi_train_data(
                args,
                training_args,
                split=(train_idxs, valid_idxs)
            )

            if args.cmplx:
                Net = UNetIGCNCmplx
            else:
                Net = UNetIGCN
            model = Net(
                n_channels=1,
                n_classes=1,
                save_dir='models/seg/isbi',
                name=f'isbi_kernel_size={training_args.kernel_size}_no_g={training_args.no_g}_base_channels={training_args.base_channels}',
                no_g=training_args.no_g,
                kernel_size=training_args.kernel_size,
                base_channels=training_args.base_channels
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
                produce_output(training_args, model=model, device=device)
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
