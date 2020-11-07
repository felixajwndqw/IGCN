import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from igcn.seg.covariance.models import IGCNCovar
from igcn.seg.covariance.loss import SegRegLoss
from igcn.seg.covariance.metrics import SegRegMetric
from quicktorch.utils import train, evaluate, imshow, get_splits
from data import CirrusDataset, TensorList
from utils import ExperimentParser, calculate_error


def write_results(**kwargs):
    sorted_keys = sorted([key for key in kwargs.keys()])
    result = '\n' + '\t'.join([str(kwargs[key]) for key in sorted_keys])

    f = open("cirrus_results.txt", "a+")
    f.write(result)
    f.close()


def collate_segreg(data):
    tensors = [
        torch.Tensor(len(data), *t.size()) for t in data[0]
    ]
    for i, d in enumerate(data):
        tensors[0][i] = d[0]
        tensors[1][i] = d[1]
        tensors[2][i] = d[2]
    return torch.tensor(tensors[0]), TensorList((tensors[1], tensors[2]))


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
    parser.add_argument('--dir',
                        default='data/stars_bad_columns_ccd', type=str,
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
        train_loader = DataLoader(
            CirrusDataset(
                os.path.join(data_dir, 'train'),
                indices=split[0],
                denoise=args.denoise,
                angle=True),
            batch_size=4, shuffle=True,
            collate_fn=collate_segreg)
        val_loader = DataLoader(
            CirrusDataset(
                os.path.join(data_dir, 'train'),
                indices=split[1],
                denoise=args.denoise,
                angle=True),
            batch_size=4, shuffle=True,
            collate_fn=collate_segreg)
        test_loader = DataLoader(
            CirrusDataset(
                os.path.join(data_dir, 'test'),
                denoise=args.denoise,
                angle=True),
            batch_size=4, shuffle=True)

        dataset = os.path.split(args.dir)[-1]
        model = IGCNCovar(
            name=f'igcn_dataset={dataset}_denoise={args.denoise}',
            n_channels=1,
            base_channels=4,
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
        criterion = SegRegLoss()
        metrics = SegRegMetric()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, training_args.lr_decay)
        m = train(model, [train_loader, val_loader], save_best=True,
                  epochs=training_args.epochs, opt=optimizer, device=device,
                  criterion=criterion,
                  metrics=metrics,
                  sch=scheduler)

        print('Evaluating')
        temp_metrics = evaluate(model, test_loader, device=device, report_iou=not args.denoise)
        m['accuracy'] = temp_metrics['accuracy']
        m['precision'] = temp_metrics['precision']
        m['recall'] = temp_metrics['recall']
        m['iou'] = temp_metrics['iou']
        del(model)
        torch.cuda.empty_cache()
        metrics.append(m)

    mean_m = {key: sum(mi[key] for mi in metrics) / training_args.nsplits for key in m.keys()}
    best_acc = max([mi['accuracy'] for mi in metrics])
    best_split = [mi['accuracy'] for mi in metrics].index(best_acc) + 1
    best_iou = metrics[[mi['accuracy'] for mi in metrics].index(best_acc)]['iou']
    mean_m['epoch'] = metrics[best_split-1]['epoch']
    error_m = {'accuracy': 0}
    if training_args.nsplits > 1:
        error_m = {key: calculate_error([mi[key] for mi in metrics])
                   for key in m.keys()}
    write_results(
        **{
            'params': total_params,
            'standard': args.standard,
            'psnr': mean_m['accuracy'],
            'psnr_error': error_m['accuracy'],
            'best_acc': best_acc,
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
