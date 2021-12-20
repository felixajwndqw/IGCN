import albumentations
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from igcn.seg.models import UNetIGCNCmplx
from igcn.seg.attention.models import DAFMS, DAFStackSmall
from quicktorch.utils import train, evaluate, get_splits
from quicktorch.metrics import MetricTracker, SegmentationTracker, MCMLSegmentationTracker
from quicktorch.writers import LabScribeWriter
from quicktorch.modules.attention.loss import (
    FocalWithLogitsLoss,
    UnifiedFocalWithLogitsLoss,
    AsymmetricFocalTvesrkyWithLogitsLoss
)
from igcn.seg.attention.metrics import DAFMetric
from cirrus.data import CirrusDataset, LSBDataset
from experiment_utils import ExperimentParser, calculate_error
from unet import UNetCmplx
from segmentation import get_metrics_criterion, create_model, load
from myvis.vis import visualise_attention


datasets = {
    'cirrus': CirrusDataset,
    'lsb': LSBDataset
}


def get_train_data(training_args, args, split):
    Dataset = datasets[args.dataset]
    trainloader = DataLoader(
        Dataset(
            survey_dir=args.survey_dir,
            mask_dir=args.mask_dir,
            transform=albumentations.Compose([
                albumentations.RandomCrop((args.img_size) * args.downscale, (args.img_size) * args.downscale),
                albumentations.Resize((args.img_size), (args.img_size)),
                # albumentations.RandomBrightnessContrast(),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.PadIfNeeded(args.img_size + args.padding, args.img_size + args.padding, border_mode=4)
            ]),
            indices=split[0],
            bands=args.bands,
            aug_mult=max(1, 512 // args.img_size),
            padding=args.padding,
            class_map=args.class_map,
            keep_background=args.background_class,
        ),
        batch_size=training_args.batch_size,
        shuffle=True,
        # num_workers=2,
        pin_memory=True,
    )
    validloader = DataLoader(
        Dataset(
            survey_dir=args.survey_dir,
            mask_dir=args.mask_dir,
            transform=albumentations.Compose([
                albumentations.RandomCrop((args.img_size) * args.downscale, (args.img_size) * args.downscale),
                albumentations.Resize((args.img_size), (args.img_size)),
                # albumentations.RandomBrightnessContrast(),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.PadIfNeeded(args.img_size + args.padding, args.img_size + args.padding, border_mode=4)
            ]),
            indices=split[1],
            bands=args.bands,
            aug_mult=max(1, 1024 // args.img_size),
            padding=args.padding,
            class_map=args.class_map,
            keep_background=args.background_class,
        ),
        batch_size=training_args.batch_size,
        shuffle=True,
        # num_workers=2,
        pin_memory=True,
    )
    return trainloader, validloader


def get_test_data(training_args, args, test_idxs):
    Dataset = datasets[args.dataset]
    testloader = DataLoader(
        Dataset(
            survey_dir=args.survey_dir,
            mask_dir=args.mask_dir,
            transform=albumentations.Compose([
                albumentations.RandomCrop((args.img_size) * args.downscale, (args.img_size) * args.downscale),
                albumentations.Resize((args.img_size), (args.img_size)),
                albumentations.PadIfNeeded(args.img_size + args.padding, args.img_size + args.padding, border_mode=4)
            ]),
            indices=test_idxs,
            bands=args.bands,
            aug_mult=max(1, 1024 // args.img_size),
            padding=args.padding,
            class_map=args.class_map,
            keep_background=args.background_class,
        ),
        batch_size=training_args.batch_size,
        shuffle=True
    )
    return testloader


def write_results(**kwargs):
    sorted_keys = sorted([key for key in kwargs.keys()])
    result = '\n' + '\t'.join([str(kwargs[key]) for key in sorted_keys])

    f = open("cirrus_results.txt", "a+")
    f.write(result)
    f.close()


seg_losses = {
    'bce': nn.BCEWithLogitsLoss,
    'unified': UnifiedFocalWithLogitsLoss,
    'focaltversky': AsymmetricFocalTvesrkyWithLogitsLoss,
    'focal': FocalWithLogitsLoss,
}


def run_cirrus_split(net_args, training_args, args, writer=None,
                     device='cuda:0', split=None, split_no=0):
    train_data, valid_data = get_train_data(
        training_args,
        args,
        split=split,
    )

    model = create_model(
        args.save_dir,
        variant=args.model_variant,
        n_channels=len(args.bands),
        n_classes=args.n_classes,
        bands=args.bands,
        downscale=args.downscale,
        model_path=args.model_path,
        padding=args.padding,
        class_map=args.class_map,
        name_params={
            'consensus': os.path.split(args.mask_dir)[-1],
            'loss': args.loss_type,
        },
        **vars(net_args)
    ).to(device)


    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    total_params = total_params / 1000000
    print("Total # parameter: " + str(total_params) + "M")

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_args.lr,
        weight_decay=training_args.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        training_args.lr_decay
    )
    class_balances = train_data.dataset.class_balances
    pos_weight = torch.sqrt(torch.tensor(class_balances, device=device))
    pos_weight = pos_weight.view(pos_weight.shape[0], 1, 1)
    seg_criterion = seg_losses[args.loss_type](reduction='none', pos_weight=pos_weight)

    metrics_class, criterion = get_metrics_criterion(
        args.model_variant,
        n_classes=args.n_classes,
        lsb=net_args.dataset == 'lsb',
        pos_weight=pos_weight,
        seg_criterion=seg_criterion
    )

    metrics_class.Writer = writer

    start = time.time()
    m = train(
        model,
        [train_data, valid_data],
        criterion=criterion,
        save_best=True,
        epochs=training_args.epochs,
        opt=optimizer,
        device=device,
        sch=scheduler,
        metrics=metrics_class,
        val_epochs=5
    )

    time_taken = time.time() - start
    mins = int(time_taken // 60)
    secs = int(time_taken % 60)
    del(model)
    torch.cuda.empty_cache()
    stats = {
        'mins': mins,
        'secs': secs,
        'total_params': total_params
    }
    return m, stats


def run_evaluation_split(net_args, training_args, args, model_path, test_idxs,
                         device='cuda:0'):
    test_data = get_test_data(training_args, args, test_idxs)

    eval_model = create_model(
        args.save_dir,
        variant=args.model_variant,
        n_channels=len(args.bands),
        n_classes=args.n_classes,
        bands=args.bands,
        downscale=args.downscale,
        padding=args.padding,
        class_map=args.class_map,
        **vars(net_args),
    ).to(device)
    load(eval_model, model_path, legacy=False, pretrain=False)

    if 'DAF' in args.model_variant:
        metrics_class = DAFMetric(full_metrics=True, n_classes=args.n_classes)
    else:
        if args.n_classes > 1:
            metrics_class = MCMLSegmentationTracker(full_metrics=True, n_classes=args.n_classes)
        else:
            metrics_class = SegmentationTracker(full_metrics=True)
    eval_m = evaluate(
        eval_model,
        test_data,
        device=device,
        metrics=metrics_class
    )

    return eval_m


def visualise(net_args, training_args, args, model_path,
              device='cuda:0'):
    dataset = datasets[net_args.dataset](
        survey_dir=args.survey_dir,
        mask_dir=args.mask_dir,
        transform=albumentations.Compose([
            albumentations.RandomCrop((args.img_size) * args.downscale, (args.img_size) * args.downscale),
            albumentations.Resize((args.img_size), (args.img_size)),
            albumentations.PadIfNeeded(args.img_size + args.padding, args.img_size + args.padding, border_mode=4)
        ]),
        bands=args.bands,
        aug_mult=1,
        padding=args.padding,
        class_map=args.class_map,
        keep_background=args.background_class,
    )

    model = create_model(
        args.save_dir,
        variant=args.model_variant,
        n_channels=len(args.bands),
        n_classes=args.n_classes,
        bands=args.bands,
        downscale=args.downscale,
        padding=args.padding,
        class_map=args.class_map,
        **vars(net_args),
    ).to(device)
    load(model, model_path, legacy=False, pretrain=False)

    img, mask = dataset.get_galaxy(args.galaxy)
    img = img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    if 'DAF' in args.model_variant:
        visualise_attention(model, img, mask, torch.tensor([512, 512]))


def generate_splits(N, nsplits=1, val_ratio=0.2, test_ratio=0.15):
    test_N = int(N * test_ratio)
    test_idxs = list(range(N - test_N, N))

    splits = get_splits(N - test_N, max(int(1 / val_ratio), nsplits))  # Divide into 6 or more blocks

    return splits, test_idxs


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
    parser.n_parser.set_defaults(dataset='cirrus')
    parser.add_argument('--survey_dir',
                        default='../data/matlas_reprocessed_cirrus', type=str,
                        help='Path to survey directory. (default: %(default)s)')
    parser.add_argument('--mask_dir',
                        default='../data/cirrus', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--save_dir',
                        default='models/seg/cirrus', type=str,
                        help='Directory to save models to. (default: %(default)s)')
    parser.add_argument('--model_path',
                        default='', type=str,
                        help='Path to model, enabling pretraining/evaluation. (default: %(default)s)')
    parser.add_argument('--model_variant',
                        default='SFC', type=str,
                        choices=[
                            'SFC', 'SFCT', 'SFCP', 'SFCReal',
                            'DAF', 'DAFT', 'DAFP',
                            'DAFMS', 'DAFMST', 'DAFMSP',
                            'DAFMSPlain', 'DAFMSPlainT', 'DAFMSPlainP',
                            'Standard', 'StandardT', 'StandardP',
                        ], help='Model variant. (default: %(default)s)')
    parser.add_argument('--n_classes',
                        default=1, type=int,
                        help='Number of classes to predict. '
                             '(default: %(default)s)')
    parser.add_argument('--bands',
                        default=['g'], type=str, nargs='+',
                        help='Image wavelength band to train on. '
                             '(default: %(default)s)')
    parser.add_argument('--downscale',
                        default=1, type=int,
                        help='Ratio to downscale images by. '
                             '(default: %(default)s)')
    parser.add_argument('--evaluate',
                        default=False, action='store_true',
                        help='Evaluates given model path.')
    parser.add_argument('--img_size',
                        default=1024, type=int,
                        help='Image size (after downscaling) network should be trained on.')
    parser.add_argument('--padding',
                        default=32, type=int,
                        help='Amount to pad images by for training.')
    parser.add_argument('--class_map',
                        default=None,
                        choices=[None, *CirrusDataset.class_maps.keys()],
                        help='Which class map to use. (default: %(default)s)')
    parser.add_argument('--loss_type',
                        default='bce',
                        choices=['bce', 'unified', 'focaltversky'],
                        help='Which class map to use. (default: %(default)s)')
    parser.add_argument('--background_class',
                        default=False, action='store_true',
                        help='Evaluates given model path.')
    parser.add_argument('--visualise',
                        default=False, action='store_true',
                        help='Visualises network outputs for given galaxy.')
    parser.add_argument('--galaxy',
                        default='NGC1121', type=str,
                        help='Which galaxy to visualise. (default: %(default)s)')

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metrics = []
    save_paths = []
    N = datasets[net_args.dataset].get_N(
        args.survey_dir,
        args.mask_dir,
        args.bands,
        class_map=args.class_map
    )
    splits, test_idxs = generate_splits(N, training_args.nsplits)

    if args.class_map is not None:
        args.n_classes = len(datasets[net_args.dataset].class_maps[args.class_map]['classes']) - 1

    if args.evaluate:
        eval_m = run_evaluation_split(
            net_args,
            training_args,
            args,
            args.model_path,
            test_idxs,
            device='cuda:0',
        )
        return

    if args.visualise:
        visualise(
            net_args,
            training_args,
            args,
            args.model_path,
            device='cuda:0',
        )
        return

    exp_name = (
        f'{args.model_variant}-Cirrus'
        f'-pre={bool(args.model_path)}'
        f'-base_channels={net_args.base_channels}'
        f'-no_g={net_args.no_g}'
        f'-bands={args.bands}'
        f'-downscale={args.downscale}'
        f'-gp={args.final_gp}'
        f'-relu={args.relu_type}'
    )
    writer = LabScribeWriter(
        'Results',
        exp_name=exp_name,
        exp_worksheet_name='CirrusSeg',
        metrics_worksheet_name='CirrusSegMetrics',
        nsplits=training_args.nsplits
    )
    writer.begin_experiment(exp_name)
    for split_no, split in zip(range(training_args.nsplits), splits):
        print('Beginning split #{}/{}'.format(split_no + 1, training_args.nsplits))
        m, stats = run_cirrus_split(
            net_args,
            training_args,
            args,
            writer=writer,
            device=device,
            split=split
        )
        save_paths.append(m.pop('save_path'))

        if not training_args.eval_best:
            m = run_evaluation_split(
                net_args,
                training_args,
                args,
                save_paths[-1],
                test_idxs,
                device='cuda:0',
            )

        metrics.append(m)
        writer.upload_split({k: str(m[k]) for k in ('IoU',)})

    # mean_m = {key: sum(mi[key] for mi in metrics) / training_args.nsplits for key in m.keys()}
    def avg(x, n_classes):
        if n_classes > 1:
            return sum(x) / args.n_classes
        else:
            return x

    mean_m = {key: sum([avg(mi[key], args.n_classes) for mi in metrics]) / training_args.nsplits
        for key in ('IoU',)}
    best_iou = max([avg(mi['IoU'], args.n_classes) for mi in metrics])
    best_split = [avg(mi['IoU'], args.n_classes) for mi in metrics].index(best_iou) + 1
    # best_psnr = metrics[[mi['IoU'] for mi in metrics].index(best_iou)]['PSNR']
    # mean_m['epoch'] = metrics[best_split-1]['epoch']

    error_m = {'e_psnr': 0}
    if training_args.nsplits > 1:
        error_m = {f'e_{key}': calculate_error([avg(mi[key], args.n_classes) for mi in metrics])
                for key in ('IoU',)}

    if training_args.eval_best:
        eval_m = run_evaluation_split(
            net_args,
            training_args,
            args,
            save_paths[best_split-1],
            test_idxs,
            device='cuda:0',
        )
    else:
        eval_m = metrics[best_split-1]
    print(metrics)
    print(eval_m)

    total_params = stats['total_params']
    mins = stats['mins']
    secs = stats['secs']
    writer.upload_best_split(
        {
            'test_iou': str(eval_m['IoU']),
            'mean_iou': str(mean_m['IoU']),
            'best_split': best_split,
        },
        best_split
    )

    write_results(
        **{
            'params': total_params,
            # 'm_psnr': mean_m['PSNR'],
            'm_iou': mean_m['IoU'],
            'best_iou': best_iou,
            # 'best_psnr': best_psnr,
            'best_split': best_split,
            'time_split': "{:3d}m{:2d}s".format(mins, secs),
            'zpaths': save_paths,
        },
        **error_m,
        **eval_m,
        **vars(training_args),
        **vars(net_args)
    )


if __name__ == '__main__':
    main()
