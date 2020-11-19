import os
import torch
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from igcn.seg.cmplxmodels import UNetIGCNCmplx
from quicktorch.utils import train, evaluate, imshow, get_splits
from data import CirrusDataset
from utils import ExperimentParser, calculate_error
from unet import UNetCmplx
from labscribe import upload_results


def write_results(**kwargs):
    sorted_keys = sorted([key for key in kwargs.keys()])
    result = '\n' + '\t'.join([str(kwargs[key]) for key in sorted_keys])

    f = open("cirrus_results.txt", "a+")
    f.write(result)
    f.close()


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
    parser.add_argument('--dir',
                        default='data/cirrus300/constant', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--standard',
                        default=False, action='store_true',
                        help='Uses a standard UNet')
    parser.add_argument('--denoise',
                        default=False, action='store_true',
                        help='Attempts to denoise the image')

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()
    data_dir = args.dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metrics = []
    if training_args.nsplits == 1:
        splits = [[None, None]]
    else:
        splits = get_splits(200, max(6, training_args.nsplits))  # Divide into 6 or more blocks
    for split_no, split in zip(range(training_args.nsplits), splits):
        print('Beginning split #{}/{}'.format(split_no + 1, training_args.nsplits))
        print(data_dir)
        train_loader = DataLoader(CirrusDataset(os.path.join(data_dir, 'train'), indices=split[0], denoise=args.denoise),
                                  batch_size=4, shuffle=True)
        val_loader = DataLoader(CirrusDataset(os.path.join(data_dir, 'train'), indices=split[1], denoise=args.denoise),
                                batch_size=4, shuffle=True)
        test_loader = DataLoader(CirrusDataset(os.path.join(data_dir, 'test'), denoise=args.denoise),
                                 batch_size=4, shuffle=True)

        dataset = os.path.split(args.dir)[-1]
        if args.standard:
            model = UNetCmplx(
                name=f'cnn_dataset={dataset}_denoise={args.denoise}',
                n_channels=1,
                base_channels=5,
                n_classes=1,
                pooling=args.pooling
            ).to(device)
        else:
            model = UNetIGCNCmplx(
                name=f'igcn_dataset={dataset}_denoise={args.denoise}',
                n_channels=1,
                base_channels=16,
                no_g=8,
                n_classes=1,
                gp=None,
                pooling=args.pooling
            ).to(device)

        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        total_params = total_params/1000000
        print("Total # parameter: " + str(total_params) + "M")

        optimizer = optim.Adam(model.parameters(),
                               lr=training_args.lr,
                               weight_decay=training_args.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, training_args.lr_decay)
        m = train(model, [train_loader, val_loader], save_best=True,
                  epochs=training_args.epochs, opt=optimizer, device=device,
                  sch=scheduler)

        print('Evaluating')
        m = evaluate(model, test_loader, device=device)
        model_name = model.name
        del(model)
        torch.cuda.empty_cache()
        metrics.append(m)

    mean_m = {f'{key}_mean': sum(mi[key] for mi in metrics) / training_args.nsplits for key in m.keys()}
    master_key = 'PSNR' if args.denoise else 'IoU'
    best_acc = max([mi[master_key] for mi in metrics])
    best_split = [mi[master_key] for mi in metrics].index(best_acc) + 1
    best_split_metrics = metrics[best_split-1]
    error_m = {f'{key}_error': calculate_error([mi[key] for mi in metrics])
               for key in m.keys()}
    results = OrderedDict([
            *mean_m.items(),
            *best_split_metrics.items(),
            *error_m.items(),
            ('best_split', best_split),
            ('no_splits', training_args.nsplits),
            ('params', total_params),
            ('dataset', dataset),
            ('standard', args.standard),
            ('denoise', args.denoise),
    ])
    print(results)
    write_results(
        **results
        # **{
        #     'params': total_params,
        #     'standard': args.standard,
        #     'psnr': mean_m[master_key],
        #     'best_split': best_split,
        #     'dataset': dataset,
        #     'denoise': args.denoise,
        # },
        # **best_split_metrics,
        # **error_m
        # **vars(training_args)
    )
    task = 'Denoising' if args.denoise else 'Segmentation'
    upload_results(
        'Results',
        model_name,
        results,
        worksheet_name=f'Cirrus{task}'
        )
    # example = iter(test_loader).next()
    # print(example[0][0].size())
    # test_out = model(example[0].to(device))
    # imshow(torch.stack([example[0][0].cpu().detach(),
    #                     example[1][0].cpu().detach(),
    #                     torch.clamp(test_out[0], min=0, max=1).cpu().detach()]))


if __name__ == '__main__':
    main()
