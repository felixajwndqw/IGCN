import math
import time
import torch
import torch.optim as optim
import albumentations
from igcn.seg.models import UNetIGCN
from igcn.seg.cmplxmodels import UNetIGCNCmplx
from quicktorch.utils import train, evaluate, imshow, get_splits
from quicktorch.data import bsd
from utils import ExperimentParser


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
    parser.add_argument('--produce',
                        default=False, action='store_true',
                        help='Generates segmentations of test data and saves. (default: %(default)s)')
    parser.add_argument('--test',
                        default=False, action='store_true')
    parser.add_argument('--path',
                        default=None, type=str,
                        help='Path to trained model.')
    parser.add_argument('--full',
                        default=False, action='store_true',
                        help='Whether to run full test list. (default: %(default)s)')
    args = parser.parse_args()
    net_args, training_args = parser.parse_group_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.produce:
        if args.path is None:
            raise TypeError('Empty --path argument.')
        produce_output(path=args.path, batch_size=training_args.batch_size, device=device)
    else:
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
                transform=albumentations.Compose([
                    albumentations.PadIfNeeded(336, 512)
                ]),
                split=split,
                num_workers=4,
                batch_size=training_args.batch_size,
                test=True
            )

            if net_args.cmplx:
                Net = UNetIGCNCmplx
            else:
                Net = UNetIGCN
            model = Net(
                n_channels=3,
                n_classes=1,
                save_dir='models/seg/bsd',
                name=('bsd_' + parser.args_to_str(net_args)) + '_epoch87',
                **vars(net_args)
            ).to(device)

            if args.test:
                model.load()
                img_batch, mask_batch = next(iter(eval_loader))
                img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)
                out_batch = model(img_batch)
                with torch.no_grad():
                    out_batch = out_batch - out_batch.min() / (out_batch.max() - out_batch.min())
                mask_batch = mask_batch.repeat(1, 3, 1, 1)
                out_batch = out_batch.repeat(1, 3, 1, 1)
                img_batch = crop(img_batch, 481, 321)
                mask_batch = crop(mask_batch, 481, 321)
                out_batch = crop(out_batch, 481, 321)
                print(img_batch.size())
                print(mask_batch.size())
                print(out_batch.size())
                print(torch.cat([img_batch, mask_batch, out_batch], dim=0).size())
                imshow(torch.cat([img_batch, mask_batch, out_batch], dim=0))
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
