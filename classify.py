import argparse
import math
import time
import torch
import torch.optim as optim
from quicktorch.utils import train, evaluate, get_splits
from quicktorch.data import mnist, cifar, mnistrot
from igcn.models import IGCNNew


SIZES = {
    'mnist': 60000,
    'mnistrotated': 60000,
    'mnistrot': 12000,
    'mnistrp': 12000,
    'cifar': 50000,
}


def write_results(dset, kernel_size, no_g, base_channels,
                  m, no_epochs,
                  total_params, mins, secs,
                  inter_mg=False, final_mg=False, cmplx=False,
                  single=False, dropout=0.3, pooling='maxmag',
                  nfc=2, weight_init=None,
                  best_split=1, splits=5, error_m=None):
    if dset == 'mnistrot':  # this is dumb but it works with my dumb notation
        dset = 'mnistr'
    if pooling == 'maxmag':
        pooling = 'mag'
    f = open("results.txt", "a+")
    out = ("\n" + dset +
           "\t" + str(kernel_size) +
           "\t\t" + str(no_g) +
           "\t\t" + str(base_channels) +
           '\t\t' + "{:1.2f}".format(dropout) +
           "\t" + str(inter_mg) +
           "\t" + str(final_mg) +
           "\t" + str(cmplx) +
           "\t" + str(single) +
           '\t' + str(pooling) +
           '\t\t' + str(nfc) +
           '\t' + str(weight_init)[:2] +
           '\t' + "{:1.4f}".format(m['accuracy']) +
           "\t" + "{:1.4f}".format(m['precision']) +
           "\t" + "{:1.4f}".format(m['recall']) +
           "\t" + str(m['epoch']) +
           "\t\t" + str(no_epochs) +
           "\t\t" + str(best_split) +
           "\t\t" + str(splits) +
           '\t\t' + "{:1.4f}".format(total_params) +
           '\t' + "{:3d}m{:2d}s".format(mins, secs))
    if error_m is not None:
        out += (
           '\t' + "{:1.4f}".format(error_m['accuracy']) +
           "\t" + "{:1.4f}".format(error_m['precision']) +
           "\t" + "{:1.4f}".format(error_m['recall'])
        )
    f.write(out)
    f.close()


def run_exp(dset, kernel_size, base_channels, no_g, dropout,
            inter_mg, final_mg, cmplx, single, pooling, nfc, weight_init,
            no_epochs=250, lr=1e-4, weight_decay=1e-7, device='0', nsplits=1):
    metrics = []
    if nsplits == 1:
        splits = [None]
    else:
        splits = get_splits(SIZES[dset], max(6, nsplits))  # Divide into 6 or more blocks
    for split_no, split in enumerate(splits):
        print('Beginning split #{}/{}'.format(split_no + 1, nsplits))
        n_channels = 1
        n_classes = 10
        if 'mnist' in dset:
            b_size = 4096 // (base_channels)
            if cmplx:
                b_size //= 2
            if dset == 'mnist':
                train_loader, test_loader, _ = mnist(batch_size=b_size,
                                                     rotate=False,
                                                     num_workers=4)
            if dset == 'mnistrotated':
                train_loader, test_loader, _ = mnist(batch_size=b_size,
                                                     rotate=True,
                                                     num_workers=4)
            if dset == 'mnistrot':
                train_loader, test_loader, _ = mnistrot(batch_size=b_size,
                                                        num_workers=4, split=split)
            if dset == 'mnistrp':
                train_loader, test_loader, _ = mnistrot(batch_size=b_size,
                                                        num_workers=4, split=split,
                                                        rotate=True)
        if dset == 'cifar':
            train_loader, test_loader, _ = cifar(batch_size=2048)
            n_channels = 3
        if dset == 'cifar100':
            train_loader, test_loader, _ = cifar(batch_size=2048, hundred=True)
            n_channels = 3
            n_classes = 100

        model = IGCNNew(no_g=no_g, n_channels=n_channels, n_classes=n_classes,
                        base_channels=base_channels, kernel_size=kernel_size,
                        inter_mg=inter_mg, final_mg=final_mg,
                        cmplx=cmplx, pooling=pooling, single=single,
                        dropout=dropout, nfc=nfc,
                        weight_init=weight_init,
                        dset=dset).to(device)

        print("Training {}".format(model.name))

        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        total_params = total_params/1000000
        print("Total # parameter: " + str(total_params) + "M")

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler = None
        start = time.time()
        m = train(model, [train_loader, test_loader], save_best=True,
                  epochs=no_epochs, opt=optimizer, device=device,
                  sch=scheduler)

        time_taken = time.time() - start
        mins = int(time_taken // 60)
        secs = int(time_taken % 60)

        if dset == 'mnistrot' or dset == 'mnistrp':
            eval_loader, _ = mnistrot(batch_size=b_size,
                                      num_workers=4,
                                      test=True)
            print('Evaluating')
            temp_metrics = evaluate(model, eval_loader, device=device)
            m['accuracy'] = temp_metrics['accuracy']
            m['precision'] = temp_metrics['precision']
            m['recall'] = temp_metrics['recall']
        del(model)
        torch.cuda.empty_cache()
        metrics.append(m)

    mean_m = {key: sum(mi[key] for mi in metrics) / nsplits for key in m.keys()}
    best_acc = max([mi['accuracy'] for mi in metrics])
    best_split = [mi['accuracy'] for mi in metrics].index(best_acc) + 1
    mean_m['epoch'] = metrics[best_split-1]['epoch']
    error_m = None
    if nsplits > 1:
        error_m = {key: math.sqrt(sum((mi[key] - mean_m[key]) ** 2 for mi in metrics) / (nsplits * (nsplits - 1)))
                   for key in m.keys()}

    write_results(dset, kernel_size, no_g, base_channels,
                  mean_m, no_epochs,
                  total_params, mins, secs,
                  inter_mg=inter_mg, final_mg=final_mg, cmplx=cmplx,
                  single=single, dropout=dropout, pooling=pooling,
                  nfc=nfc, weight_init=weight_init,
                  best_split=best_split, splits=nsplits, error_m=error_m)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Handles MNIST/CIFAR tasks.')

    parser.add_argument('--dataset',
                        default='mnistrot', type=str,
                        choices=['mnist', 'mnistrotated', 'mnistrot', 'mnistrp', 'cifar'],
                        help='Type of dataset. Choices: %(choices)s (default: %(default)s)')
    parser.add_argument('--kernel_size',
                        default=3, type=int,
                        help='Kernel size')
    parser.add_argument('--base_channels',
                        default=16, type=int,
                        help='Number of feature channels in first layer of network.')
    parser.add_argument('--no_g',
                        default=4, type=int,
                        help='Number of Gabor filters.')
    parser.add_argument('--dropout',
                        default=0.3, type=float,
                        help='Learning rate.')
    parser.add_argument('--inter_mg',
                        default=False, action='store_true',
                        help='Whether to pool over orientations in intermediate layers. (default: %(default)s)')
    parser.add_argument('--final_mg',
                        default=False, action='store_true',
                        help='Whether to pool over orientations in final layers. (default: %(default)s)')
    parser.add_argument('--cmplx',
                        default=False, action='store_true',
                        help='Whether to use a complex architecture.')
    parser.add_argument('--single',
                        default=False, action='store_true',
                        help='Whether to use a single gconv layer between each pooling layer.')
    parser.add_argument('--pooling',
                        default='maxmag', type=str,
                        choices=['max', 'maxmag', 'avg'],
                        help='Type of pooling. Choices: %(choices)s (default: %(default)s)')
    parser.add_argument('--nfc',
                        default=2, type=int,
                        help='Number of fully connected layers before classification.')
    parser.add_argument('--weight_init',
                        default=None,
                        type=lambda x: None if x == 'None' else x,
                        choices=[None, 'he', 'glorot'],
                        help=('Type of weight initialisation. Choices: %(choices)s '
                              '(default: %(default)s, corresponding to re/im independent He init.)'))

    parser.add_argument('--epochs',
                        default=250, type=int,
                        help='Number of epochs to train over.')
    parser.add_argument('--lr',
                        default=1e-4, type=float,
                        help='Learning rate.')
    parser.add_argument('--weight_decay',
                        default=1e-7, type=float,
                        help='Weight decay/l2 reg.')
    parser.add_argument('--splits',
                        default=1, type=int,
                        help='Number of validation splits to train over')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_exp(
        args.dataset,
        args.kernel_size,
        args.base_channels,
        args.no_g,
        args.dropout,
        args.inter_mg,
        args.final_mg,
        args.cmplx,
        args.single,
        args.pooling,
        args.nfc,
        args.weight_init,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
        nsplits=args.splits
    )


if __name__ == '__main__':
    main()
