from igcn.seg.attention.loss import DAFLoss
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from igcn.seg.cmplxmodels import UNetIGCNCmplx
from igcn.seg.attention.models import DAFStackSmall
from igcn.seg.attention.metrics import DAFMetric
from quicktorch.utils import train, evaluate, imshow, get_splits
from cirrus.data import SynthCirrusDataset
from utils import ExperimentParser, calculate_error
from unet import UNetCmplx
import albumentations


SIZES = {
    'stars_bad_columns_ccd': 300,
    'stars_bad_columns_ccd_saturation': 1200
}


def write_results(**kwargs):
    sorted_keys = sorted([key for key in kwargs.keys()])
    result = '\n' + '\t'.join([str(kwargs[key]) for key in sorted_keys])
    f = open("cirrus_results.txt", "a+")
    f.write(result)
    f.close()


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
    parser.add_argument('--dir',
                        default='../data/stars_bad_columns_ccd', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--model_variant',
                        default="SFC", type=str,
                        choices=['SFC', 'Standard', 'DAF'],
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--denoise',
                        default=False, action='store_true',
                        help='Attempts to denoise the image')

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()
    data_dir = args.dir
    N = SIZES[os.path.split(args.dir)[-1]]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metrics = []
    save_paths = []
    if training_args.nsplits == 1:
        splits = [[None, None]]
    else:
        splits = get_splits(N // 3 * 2, max(6, training_args.nsplits))  # Divide into 6 or more blocks
    for split_no, split in zip(range(training_args.nsplits), splits):
        print('Beginning split #{}/{}'.format(split_no + 1, training_args.nsplits))
        train_loader = DataLoader(
            SynthCirrusDataset(
                os.path.join(data_dir, 'train'),
                indices=split[0],
                denoise=args.denoise,
                transform=albumentations.Compose([
                    albumentations.RandomCrop(256, 256),
                    albumentations.Flip(),
                    albumentations.RandomRotate90(),
                    albumentations.PadIfNeeded(288, 288, border_mode=4)
                ]),
            ),
            batch_size=training_args.batch_size, shuffle=True)
        val_loader = DataLoader(
            SynthCirrusDataset(
                os.path.join(data_dir, 'train'),
                indices=split[1],
                denoise=args.denoise,
                transform=albumentations.Compose([
                    albumentations.RandomCrop(256, 256),
                    albumentations.Flip(),
                    albumentations.RandomRotate90(),
                    albumentations.PadIfNeeded(288, 288, border_mode=4)
                ]),
            ),
            batch_size=training_args.batch_size, shuffle=True)
        test_loader = DataLoader(
            SynthCirrusDataset(
                os.path.join(data_dir, 'test'),
                denoise=args.denoise,
                transform=albumentations.Compose([
                    albumentations.RandomCrop(256, 256),
                    albumentations.Flip(),
                    albumentations.PadIfNeeded(288, 288, border_mode=4)
                ]),
            ),
            batch_size=training_args.batch_size, shuffle=True)

        dataset = os.path.split(args.dir)[-1]
        if args.model_variant == 'Standard':
            model = UNetCmplx(
                name=f'cnn_dataset={dataset}_denoise={args.denoise}',
                n_channels=1,
                base_channels=5,
                n_classes=1,
                pooling=net_args.pooling
            ).to(device)
        elif args.model_variant == 'DAF':
            model = DAFStackSmall(
                name=f'DAFSmall_dataset={dataset}_denoise={args.denoise}',
                n_channels=1,
                base_channels=net_args.base_channels,
                no_g=net_args.no_g,
                n_classes=1,
                pooling=args.pooling
            ).to(device)
        else:
            model = UNetIGCNCmplx(
                name=f'igcn_dataset={dataset}_denoise={args.denoise}',
                n_channels=1,
                base_channels=net_args.base_channels,
                no_g=net_args.no_g,
                n_classes=1,
                pooling=net_args.pooling,
                gp=net_args.final_gp,
                relu_type=net_args.relu_type,
                mode=net_args.upsample_mode,
            ).to(device)

        if args.model_variant == 'DAF':
            MetricsClass = DAFMetric()
            criterion = DAFLoss()
        else:
            MetricsClass = None
            criterion = nn.CrossEntropyLoss()

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
                  sch=scheduler, metrics=MetricsClass, criterion=criterion)

        print('Evaluating')
        temp_metrics = evaluate(model, test_loader, device=device)
        m['PSNR'] = temp_metrics['PSNR']
        m['IoU'] = temp_metrics['IoU']
        del(model)
        save_paths.append(m.pop('save_path'))
        torch.cuda.empty_cache()
        metrics.append(m)

    mean_m = {key: sum(mi[key] for mi in metrics) / training_args.nsplits for key in m.keys()}
    best_iou = max([mi['IoU'] for mi in metrics])
    best_split = [mi['IoU'] for mi in metrics].index(best_iou) + 1
    best_psnr = metrics[[mi['IoU'] for mi in metrics].index(best_iou)]['PSNR']
    mean_m['epoch'] = metrics[best_split-1]['epoch']

    error_m = {'e_psnr': 0}
    if training_args.nsplits > 1:
        error_m = {key: calculate_error([mi[key] for mi in metrics])
                   for key in m.keys()}
    write_results(
        **{
            'params': total_params,
            'standard': args.model_variant,
            'psnr': mean_m['PSNR'],
            'psnr_error': error_m['PSNR'],
            'best_psnr': best_psnr,
            'best_iou': best_iou,
            'best_split': best_split,
            'dataset': dataset,
            'denoise': args.denoise,
        },
        # **vars(training_args)
    )
    # example = iter(test_loader).next()
    # print(example[0][0].size())
    # test_out = model(example[0].to(device))
    # imshow(torch.stack([example[0][0].cpu().detach(),
    #                     example[1][0].cpu().detach(),
    #                     torch.clamp(test_out[0], min=0, max=1).cpu().detach()]))


if __name__ == '__main__':
    main()
