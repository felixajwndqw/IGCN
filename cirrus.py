import albumentations
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from igcn.seg.cmplxmodels import UNetIGCNCmplx, CirrusIGCN
from quicktorch.utils import train, evaluate, get_splits
from quicktorch.metrics import MetricTracker
from quicktorch.writers import LabScribeWriter
from cirrus.data import CirrusDataset
from utils import ExperimentParser, calculate_error
from unet import UNetCmplx


def get_N(survey_dir, mask_dir, bands):
    # mask_paths = glob.glob(os.path.join(mask_dir, '*.npz'))
    # if set(bands) == set('g'):
    #     return 116
    # if set(bands) == set('r'):
    #     return 108
    # if set(bands) == set(['g', 'r']):
    #     return 108
    # Cirrus only 
    if set(bands) == set(['g', 'r']):
        return 48
    # Cirrus + HB
    # if set(bands) == set('g'):
    #     return 184
    # if set(bands) == set('r'):
    #     return 184
    # if set(bands) == set(['g', 'r']):
    #     return 184


def get_train_data(training_args, args, split):
    trainloader = DataLoader(
        CirrusDataset(
            survey_dir=args.survey_dir,
            mask_dir=args.mask_dir,
            transform=albumentations.Compose([
                albumentations.RandomCrop(256 * args.downscale, 256 * args.downscale),
                albumentations.Resize(256, 256),
                albumentations.RandomBrightnessContrast(),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.PadIfNeeded(288, 288, border_mode=4)
            ]),
            indices=split[0],
            bands=args.bands,
            aug_mult=4,
        ),
        batch_size=training_args.batch_size,
        shuffle=True,
        # num_workers=2,
        pin_memory=True,
    )
    validloader = DataLoader(
        CirrusDataset(
            survey_dir=args.survey_dir,
            mask_dir=args.mask_dir,
            transform=albumentations.Compose([
                albumentations.RandomCrop(256 * args.downscale, 256 * args.downscale),
                albumentations.Resize(256, 256),
                albumentations.RandomBrightnessContrast(),
                albumentations.Flip(),
                albumentations.RandomRotate90(),
                albumentations.PadIfNeeded(288, 288, border_mode=4)
            ]),
            indices=split[1],
            bands=args.bands,
            aug_mult=4,
        ),
        batch_size=training_args.batch_size,
        shuffle=True,
        # num_workers=2,
        pin_memory=True,
    )
    return trainloader, validloader


def create_model(save_dir, args, variant="SFC", n_channels=1, n_classes=2, bands=['g'], downscale=1, model_path=''):
    model_fn = UNetIGCNCmplx
    if variant == "SFCT":
        model_fn = CirrusIGCN
    if variant == "Standard":
        model_fn = UNetCmplx
    model = model_fn(
        n_channels=n_channels,
        n_classes=n_classes,
        save_dir=save_dir,
        name=f'{variant}-cirrus'
             f'_bands={bands}'
             f'-pre={bool(model_path)}'
             f'-kernel_size={args.kernel_size}'
             f'-no_g={args.no_g}'
             f'-base_channels={args.base_channels}'
             f'-downscale={downscale}',
        no_g=args.no_g,
        kernel_size=args.kernel_size,
        base_channels=args.base_channels
    )
    if model_path:
        load(model, model_path)
    return model


def write_results(**kwargs):
    sorted_keys = sorted([key for key in kwargs.keys()])
    result = '\n' + '\t'.join([str(kwargs[key]) for key in sorted_keys])

    f = open("cirrus_results.txt", "a+")
    f.write(result)
    f.close()


def load(model, save_path):
    checkpoint = torch.load(save_path)

    weight = checkpoint['model_state_dict']['inc.conv1.weight']
    checkpoint['model_state_dict']['inc.conv1.weight'] = weight.repeat(1, 1, 2, 1, 1)

    model.load_state_dict(checkpoint['model_state_dict'])


def run_cirrus_split(net_args, training_args, args, writer=None,
                     device='cuda:0', split=None, split_no=0):
    train_data, valid_data = get_train_data(
        training_args,
        args,
        split=split
    )

    # dataset = os.path.split(args.dir)[-1]
    model = create_model(
        args.save_dir,
        net_args,
        variant=args.model_variant,
        n_channels=len(args.bands),
        n_classes=args.n_classes,
        bands=args.bands,
        downscale=args.downscale,
        model_path=args.model_path,
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
    metrics_class = MetricTracker.detect_metrics(train_data)
    metrics_class.Writer = writer

    start = time.time()
    m = train(
        model,
        [train_data, valid_data],
        save_best=True,
        epochs=training_args.epochs,
        opt=optimizer,
        device=device,
        sch=scheduler,
        metrics=metrics_class
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
    eval_model = create_model(
        args.save_dir,
        net_args,
        variant=args.model_variant,
        n_channels=len(args.bands),
        n_classes=args.n_classes,
        bands=args.bands,
        downscale=args.downscale
    ).to(device)
    eval_model.load(save_path=model_path)

    test_data = DataLoader(
        CirrusDataset(
            survey_dir=args.survey_dir,
            mask_dir=args.mask_dir,
            transform=albumentations.Compose([
                albumentations.RandomCrop(256 * args.downscale, 256 * args.downscale),
                albumentations.Resize(256, 256),
                albumentations.Flip(),
                albumentations.PadIfNeeded(288, 288, border_mode=4)
            ]),
            indices=test_idxs,
            bands=args.bands,
            aug_mult=3
        ),
        batch_size=training_args.batch_size,
        shuffle=True
    )

    eval_m = evaluate(
        eval_model,
        test_data,
        device=device
    )

    return eval_m


def generate_splits(N, nsplits=1, val_ratio=0.2, test_ratio=0.15):
    test_N = int(N * test_ratio)
    test_idxs = list(range(test_N, N))

    splits = get_splits(N - test_N, max(int(1 / val_ratio), nsplits))  # Divide into 6 or more blocks

    return splits, test_idxs


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
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
                        help='Path to model, enabling pretraining. (default: %(default)s)')
    parser.add_argument('--model_variant',
                        default='SFC', type=str,
                        choices=['SFC', 'SFCT', 'Standard'],
                        help='Model variant. (default: %(default)s)')
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

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()
    print(args.bands)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metrics = []
    save_paths = []
    N = get_N(args.survey_dir, args.mask_dir, args.bands)
    splits, test_idxs = generate_splits(N, training_args.nsplits)

    if args.evaluate:
        eval_m = run_evaluation_split(
            net_args,
            training_args,
            args,
            args.model_path,
            test_idxs,
            device='cuda:0',
        )
        print(eval_m)
        return

    exp_name = (
        f'{args.model_variant}-Cirrus'
        f'-pre={bool(args.model_path)}'
        f'-base_channels={net_args.base_channels}'
        f'-no_g={net_args.no_g}'
        f'-bands={args.bands}'
        f'-downscale={args.downscale}'
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
        metrics.append(m)
        writer.upload_split({k: m[k] for k in ('IoU', 'PSNR', 'loss')})

    mean_m = {key: sum(mi[key] for mi in metrics) / training_args.nsplits for key in m.keys()}
    best_iou = max([mi['IoU'] for mi in metrics])
    best_split = [mi['IoU'] for mi in metrics].index(best_iou) + 1
    best_psnr = metrics[[mi['IoU'] for mi in metrics].index(best_iou)]['PSNR']
    mean_m['epoch'] = metrics[best_split-1]['epoch']

    error_m = {'e_psnr': 0}
    if training_args.nsplits > 1:
        error_m = {f'e_{key}': calculate_error([mi[key] for mi in metrics])
                   for key in m.keys()}

    eval_m = run_evaluation_split(
        net_args,
        training_args,
        args,
        save_paths[best_split-1],
        test_idxs,
        device='cuda:0',
    )

    total_params = stats['total_params']
    mins = stats['mins']
    secs = stats['secs']
    writer.upload_best_split(
        {
            'test_iou': eval_m['IoU'],
            'mean_iou': mean_m['IoU'],
            'best_split': best_split,
        },
        best_split
    )

    write_results(
        **{
            'params': total_params,
            'm_psnr': mean_m['PSNR'],
            'm_iou': mean_m['IoU'],
            'best_iou': best_iou,
            'best_psnr': best_psnr,
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
