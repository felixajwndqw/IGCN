import math
import glob
import os
import time
import torch
import torch.optim as optim
import albumentations
from igcn.seg.models import UNetIGCN, UNetIGCNCmplx
from quicktorch.utils import train, evaluate, imshow, get_splits
from quicktorch.data import bsd
from utils import ExperimentParser
from segmentation import create_model
from isbi import parse_filename
import torch.nn.functional as F
import torchvision

SIZE = 300


def produce_output(model=None, path=None, padding=16, batch_size=8, device='cpu'):
    pass


def crop(t, w, h):
    dw = t.size(-1) - w
    dh = t.size(-2) - h
    return t[..., dh//2:dh//2+h, dw//2:dw//2+w]


def write_results(dset='bsd', kernel_size=3, no_g=8, base_channels=16, m={},
                  no_epochs=100, total_params=1, mins=None, secs=None,
                  cmplx=False,
                  best_split=1, nsplits=1, error_m=None, **kwargs):
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
    parser.add_argument('--test',
                        default=False, action='store_true')
    parser.add_argument('--model_path',
                        default=None, type=str,
                        help='Path to trained model.')
    parser.add_argument('--results_dir',
                        default='../results/bsd/rcf', type=str,
                        help='Path to trained model.')
    parser.add_argument('--padding',
                        default=0, type=int,
                        help='Number of images used for validation dataset (only ISBI).')
    parser.add_argument('--multiscale',
                        default=False, action='store_true')
    args = parser.parse_args()
    net_args, training_args = parser.parse_group_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = []
    if training_args.nsplits == 1:
        splits = [None]
    else:
        splits = get_splits(SIZE, max(6, training_args.nsplits))
    for split_no, split in zip(range(training_args.nsplits), splits):
        trainloader, validloader = bsd(
            transform=albumentations.Compose([
                albumentations.Flip(),
                albumentations.PadIfNeeded(336, 512)
            ]),
            split=split,
            num_workers=4,
            batch_size=training_args.batch_size
        )
        eval_loader = bsd(
            # transform=albumentations.Compose([
            #     albumentations.PadIfNeeded(272 + args.padding, 272 + args.padding, border_mode=4)
            # ]),
            batch_size=1,#training_args.batch_size,
            dir='../data/bsd',
            test=True,
        )

        params = parse_filename(args.model_path)
        params['pooling'] = net_args.pooling
        params['upsample_mode'] = 'bilinear'
        params['dropout'] = 0.
        print(params)
        model = create_model(
            'models/seg/' + net_args.dataset,
            n_channels=3,
            n_classes=1,
            model_path=args.model_path,
            padding=args.padding,
            pretrain=False,
            **params,
        ).to(device)

        if args.test:
            model.eval()
            if not os.path.exists(args.results_dir):
                os.makedirs(args.results_dir)
            filenames = glob.glob('../data/bsd/processed/test/labels/*.png')
            filenames = [os.path.split(fname)[-1] for fname in filenames]
            for (img_batch, mask_batch), fname in zip(eval_loader, filenames):
                img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)
                w, h = img_batch.size(-2), img_batch.size(-1)
                outs = []
                if args.multiscale:
                    sizes, pads = [200, 400, 600], [12, 24, 36]
                else:
                    sizes, pads = [400], [24]
                for size, pad in zip(sizes, pads):
                    img_batch = F.interpolate(img_batch, (size, size), mode='bilinear')
                    img_batch = F.pad(img_batch, (pad, pad, pad, pad), 'reflect')
                    with torch.no_grad():
                        out = model(img_batch)
                    out = out[..., pad:-pad, pad:-pad]
                    out = F.interpolate(out, (w, h), mode='bilinear')
                    outs.append(out)
                out_batch = sum(outs) / len(outs)
                torchvision.utils.save_image(
                    out_batch,
                    os.path.join(args.results_dir, fname)
                )
            return

        total_params = sum(p.numel()
                            for p in model.parameters()
                            if p.requires_grad) / 1000000
        print("Total # parameter: " + str(total_params) + "M")

        optimizer = optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)

        start = time.time()
        m = train(
            model,
            [trainloader, validloader],
            epochs=training_args.epochs,
            opt=optimizer,
            device=device,
            save_best=True
        )

        time_taken = time.time() - start
        mins = int(time_taken // 60)
        secs = int(time_taken % 60)
        temp_metrics = evaluate(model, eval_loader, device=device)
        m['accuracy'] = temp_metrics['accuracy']
        del(model)
        torch.cuda.empty_cache()
        metrics.append(m)

        mean_m = {key: sum(mi[key] for mi in metrics) / training_args.nsplits for key in m.keys()}
        best_acc = max([mi['accuracy'] for mi in metrics])
        best_split = [mi['accuracy'] for mi in metrics].index(best_acc) + 1
        mean_m['epoch'] = metrics[best_split-1]['epoch']
        error_m = None
        if training_args.nsplits > 1:
            error_m = {key: math.sqrt(sum((mi[key] - mean_m[key]) ** 2 for mi in metrics) / (training_args.nsplits * (training_args.splits - 1)))
                       for key in m.keys()}
        write_results(
            'bsd',
            **vars(net_args),
            **vars(training_args),
            m=mean_m,
            total_params=total_params,
            mins=mins,
            secs=secs,
            best_split=best_split,
            error_m=error_m
        )


if __name__ == '__main__':
    main()
