import argparse
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
}


def write_results(dset, kernel_size, no_g, m, no_epochs,
                  total_params, mins, secs,
                  inter_mg=False, final_mg=False, cmplx=False):
    f = open("results.txt", "a+")
    f.write("\n" + dset +
            "\t" + str(kernel_size) +
            "\t\t" + str(no_g) +
            "\t" + str(inter_mg) +
            "\t" + str(final_mg) +
            "\t" + str(cmplx) +
            '\t' + "{:1.4f}".format(m['accuracy']) +
            "\t" + "{:1.4f}".format(m['precision']) +
            "\t" + "{:1.4f}".format(m['recall']) +
            "\t" + str(m['epoch']) +
            "\t\t" + str(no_epochs) +
            '\t\t' + "{:1.4f}".format(total_params) +
            '\t' + "{:3d}m{:2d}s".format(mins, secs))
    f.close()


def run_exp(dset, kernel_size, base_channels, no_g, inter_mg, final_mg, cmplx,
            no_epochs=250, lr=1e-4, weight_decay=1e-7, device='0', splits=1):
    splits = get_splits(SIZES[dset], splits)
    for split in splits:
        print("Training igcn{} on {}, base_channels={}, no_g={}, "
            "inter_mg={}, final_mg={}".format(kernel_size, dset, base_channels,
                                                no_g, inter_mg, final_mg))

        if dset == 'mnist' or dset == 'mnistrot':
            if inter_mg or final_mg:
                b_size = int(4096 // no_g)
            else:
                b_size = 4096
            if cmplx:
                b_size //= 4
            if dset == 'mnist':
                train_loader, test_loader, _ = mnist(batch_size=b_size,
                                                     rotate=False,
                                                     num_workers=2)
            if dset == 'mnistrotated':
                train_loader, test_loader, _ = mnist(batch_size=b_size,
                                                     rotate=True,
                                                     num_workers=2)
            if dset == 'mnistrot':
                train_loader, test_loader, _ = mnistrot(batch_size=b_size,
                                                        num_workers=2, split=split)
        if dset == 'cifar':
            train_loader, test_loader, _ = cifar(batch_size=2048)

        model = IGCNNew(no_g=no_g, kernel_size=kernel_size,
                        inter_mg=inter_mg, final_mg=final_mg, 
                        cmplx=cmplx, dset=dset).to(device)

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

        if dset == 'mnistrot':
            eval_loader, _ = mnistrot(batch_size=b_size,
                                      num_workers=8,
                                      test=True)
            print('Evaluating')
            temp_metrics = evaluate(model, eval_loader, device=device)
            m['accuracy'] = temp_metrics['accuracy']
            m['precision'] = temp_metrics['precision']
            m['recall'] = temp_metrics['recall']
        write_results(dset, kernel_size, no_g,
                    m, no_epochs,
                    total_params, mins, secs,
                    inter_mg=inter_mg, final_mg=final_mg, cmplx=cmplx)

        del(model)
        torch.cuda.empty_cache()

    return m


def main():
    parser = argparse.ArgumentParser(description='Handles MNIST/CIFAR tasks.')

    parser.add_argument('--dataset',
                        default='mnistrot', type=str,
                        choices=['mnist', 'mnistrotated', 'mnistrot', 'cifar'],
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
    parser.add_argument('--inter_mg',
                        default=False, action='store_true',
                        help='Whether to pool over orientations in intermediate layers. (default: %(default)s)')
    parser.add_argument('--final_mg',
                        default=False, action='store_true',
                        help='Whether to pool over orientations in final layers. (default: %(default)s)')
    parser.add_argument('--cmplx',
                        default=False, action='store_true',
                        help='Whether to use a complex architecture.')

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
        args.inter_mg,
        args.final_mg,
        args.cmplx,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
        splits=args.splits
    )


if __name__ == '__main__':
    main()
