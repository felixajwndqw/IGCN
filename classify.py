import time
import albumentations
import torch
import torch.nn as nn
import torch.optim as optim
import PIL
from torchvision import transforms
from quicktorch.utils import train, evaluate, get_splits
from quicktorch.data import mnist, cifar, mnistrot
from quicktorch.metrics import MetricTracker
from quicktorch.writers import LabScribeWriter
from igcn.models import IGCN, SFC, SFCNonCyclic, SFCResNet
from utils import ExperimentParser, calculate_error, AlbumentationsWrapper


SIZES = {
    'mnist': 60000,
    'mnistrotated': 60000,
    'mnistrot': 12000,
    'mnistrp': 12000,
    'cifar': 50000,
}


def run_classification_split(net_args, training_args, writer=None, device='cuda:0', split=None, split_no=0):
    n_channels = 1
    n_classes = 10
    transform = None
    if 'mnist' in net_args.dataset:
        b_size = int(128 / (net_args.no_g / 4)) // 2 # REMOVE / 2
        if training_args.batch_size != 32:
            b_size = training_args.batch_size
        if training_args.augment:
            transform = transforms.Compose([
                # transforms.Pad((0, 0, 1, 1), fill=0),
                transforms.Resize(84),
                transforms.RandomRotation(180, resample=PIL.Image.BILINEAR, expand=False),
                transforms.Resize(28),
                # AlbumentationsWrapper(albumentations.Cutout(num_holes=3))
            ])
        if net_args.dataset == 'mnist':
            train_loader, test_loader, _ = mnist(
                batch_size=b_size,
                rotate=False,
                num_workers=8
            )
        if net_args.dataset == 'mnistrotated':
            train_loader, test_loader, _ = mnist(
                batch_size=b_size,
                rotate=True,
                num_workers=8
            )
        if net_args.dataset == 'mnistrot':
            train_loader, test_loader, _ = mnistrot(
                batch_size=b_size,
                num_workers=8,
                split=split,
                transform=transform,
                onehot=False,
            )
        if net_args.dataset == 'mnistrp':
            train_loader, test_loader, _ = mnistrot(
                batch_size=b_size,
                num_workers=8,
                split=split,
                rotate=True,
                transform=transform,
                onehot=False,
            )
    if net_args.dataset == 'cifar':
        train_loader, test_loader, _ = cifar(batch_size=2048)
        n_channels = 3
    if net_args.dataset == 'cifar100':
        train_loader, test_loader, _ = cifar(batch_size=2048, hundred=True)
        n_channels = 3
        n_classes = 100

    if net_args.model_type == 'IGCN':
        model_factory = IGCN
    elif net_args.model_type == 'SFC':
        model_factory = SFC
    elif net_args.model_type == 'SFCNC':
        model_factory = SFCNonCyclic
    elif net_args.model_type == 'SFCResNet':
        model_factory = SFCResNet

    print(vars(net_args))
    model = model_factory(
        n_channels=n_channels,
        n_classes=n_classes,
        save_dir=f'models/{net_args.dataset}/',
        name=f'{net_args.model_type}{net_args.no_g}'
             f'-pool={net_args.pooling}'
             f'-gp={net_args.final_gp}'
             f'-mod={net_args.mod}',
        **vars(net_args)).to(device)
    # model = SFC(n_channels=n_channels,
    #             n_classes=n_classes,
    #             
    #             name=f'SFC{net_args.no_g}',
    #             **vars(net_args)).to(device)

    if training_args.name is not None:
        model.save_dir += 'mod/'
    model.name = f'{model.name}_split_no={split_no+1}'

    print("Training {}".format(model.name))

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = total_params/1000000
    print("Total # parameter: " + str(total_params) + "M")

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_args.lr,
        weight_decay=training_args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, training_args.lr_decay)
    # scheduler = None
    metrics_class = MetricTracker.detect_metrics(train_loader)
    metrics_class.Writer = writer

    start = time.time()
    # with torch.autograd.detect_anomaly():
    m = train(
        model, [train_loader, test_loader], save_best=True,
        epochs=training_args.epochs, opt=optimizer, criterion=criterion,
        device=device, sch=scheduler, metrics=metrics_class
    )

    time_taken = time.time() - start
    mins = int(time_taken // 60)
    secs = int(time_taken % 60)

    if net_args.dataset == 'mnistrot' or net_args.dataset == 'mnistrp':
        eval_loader, _ = mnistrot(
            batch_size=b_size,
            num_workers=4,
            test=True,
            onehot=False,
        )
        print('Evaluating')
        temp_metrics = evaluate(model, eval_loader, device=device)
        m['accuracy'] = temp_metrics['accuracy']
        m['precision'] = temp_metrics['precision']
        m['recall'] = temp_metrics['recall']
    del(model)
    torch.cuda.empty_cache()
    stats = {
        'mins': mins,
        'secs': secs,
        'total_params': total_params
    }
    return m, stats


def write_results(dataset='mnist', kernel_size=3, no_g=4, base_channels=16,
                  m={}, epochs=100,
                  total_params=1, mins=None, secs=None,
                  inter_gp=None, final_gp=None, cmplx=True,
                  single=False, dropout=0., pooling='mag',
                  nfc=2, weight_init=None, all_gp=False, bnorm='new',
                  relu_type='c', fc_type='cat', fc_block='lin',
                  fc_relu_type='c', mod='hadam',
                  best_split=1, augment=False, nsplits=5, error_m=None,
                  weight_decay=1e-7, lr=1e-4, lr_decay=1, best_acc=0,
                  translate=0, scale=0, shear=0, softmax=False,
                  l_init='uniform', sigma_init='fixed', single_param=False,
                  **kwargs):
    if dataset == 'mnistrot':  # this is dumb but it works with my dumb notation
        dataset = 'mnistr'
    if pooling == 'mag':
        pooling = 'mag'
    f = open("results.txt", "a+")
    out = ("\n" + dataset +
           "\t" + str(kernel_size) +
           "\t\t" + str(no_g) +
           "\t\t" + str(base_channels) +
           '\t\t' + "{:1.2f}".format(dropout) +
           "\t" + str(inter_gp) +
           "\t\t" + str(final_gp) +
           "\t\t" + str(relu_type) +
           "\t\t" + str(fc_type) +
           "\t\t" + str(fc_block) +
           "\t\t" + str(fc_relu_type) +
           "\t\t" + str(mod) +
           "\t\t" + str(cmplx) +
           "\t" + str(single) +
           '\t' + str(pooling) +
           '\t\t' + str(nfc) +
           '\t' + str(weight_init)[:2] +
           '\t\t' + str(all_gp) +
           '\t' + str(bnorm) +
           "\t\t" + str(softmax) +
           '\t' + str(l_init) +
           '\t' + str(sigma_init) +
           '\t' + str(single_param) +
           '\t' + "{:1.4f}".format(m['accuracy']) +
           "\t" + "{:1.4f}".format(m['precision']) +
           "\t" + "{:1.4f}".format(m['recall']) +
           "\t" + str(m['epoch']) +
           "\t\t" + str(epochs) +
           "\t\t" + str(augment) +
           '\t\t' + "{:1.2f}".format(translate) +
           "\t" + "{:1.2f}".format(scale) +
           "\t" + "{:2.2f}".format(shear) +
           "\t\t" + str(best_split) +
           '\t\t' + "{:1.4f}".format(best_acc) +
           "\t" + str(nsplits) +
           '\t\t' + "{:1.0e}".format(weight_decay) +
           '\t' + "{:1.0e}".format(lr) +
           '\t' + "{:1.0e}".format(lr_decay) +
           '\t' + "{:1.4f}".format(total_params) +
           '\t' + "{:3d}m{:2d}s".format(mins, secs))
    if error_m is not None:
        out += (
           '\t' + "{:1.4f}".format(error_m['accuracy']) +
           "\t" + "{:1.4f}".format(error_m['precision']) +
           "\t" + "{:1.4f}".format(error_m['recall'])
        )
    f.write(out)
    f.close()


def run_exp(net_args, training_args, device='0', exp_name=None, **kwargs):
    metrics = []
    metric_keys = ('accuracy', 'precision', 'recall')
    if training_args.nsplits == 1:
        splits = [None]
    else:
        splits = get_splits(SIZES[net_args.dataset], max(6, training_args.nsplits))  # Divide into 6 or more blocks

    if exp_name is None:
        exp_name = f'{net_args.model_type}-'+'-'.join([f'{key}={val}' for key, val in vars(net_args).items()])
        exp_name += '-'.join([f'{key}={val}' for key, val in vars(training_args).items()])
    writer = LabScribeWriter(
        'Results',
        exp_name=exp_name,
        exp_worksheet_name=net_args.dataset[:5].upper(),
        metrics_worksheet_name=f'{net_args.dataset[:5].upper()}Metrics',
        nsplits=training_args.nsplits
    )
    writer.begin_experiment(vars(net_args))
    for split_no, split in zip(range(training_args.nsplits), splits):
        print('Beginning split #{}/{}'.format(split_no + 1, training_args.nsplits))
        m, stats = run_classification_split(
            net_args,
            training_args,
            writer=writer,
            device=device,
            split=split,
            split_no=split_no
        )
        metrics.append(m)
        writer.upload_split({k: m[k] for k in metric_keys})

    mean_m = {key: sum(mi[key] for mi in metrics) / training_args.nsplits for key in metric_keys}
    best_acc = max([mi['accuracy'] for mi in metrics])
    best_split = [mi['accuracy'] for mi in metrics].index(best_acc) + 1
    mean_m['epoch'] = metrics[best_split-1]['epoch']
    error_m = None
    if training_args.nsplits > 1:
        error_m = {key: calculate_error([mi[key] for mi in metrics])
                   for key in metric_keys}
    writer.upload_best_split(
        {
            'best_accuracy': best_acc,
            'best_split': best_split,
            'error': None if error_m is None else error_m['accuracy']
        },
        split=best_split
    )

    total_params = stats['total_params']
    mins = stats['mins']
    secs = stats['secs']

    write_results(
        **vars(net_args),
        **vars(training_args),
        m=mean_m,
        total_params=total_params,
        mins=mins,
        secs=secs,
        best_split=best_split,
        best_acc=best_acc,
        error_m=error_m
    )

    return best_acc, mean_m['accuracy']


def main():
    parser = ExperimentParser(description='Handles MNIST/CIFAR tasks.')
    net_args, training_args = parser.parse_group_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if training_args.debug:
        run_classification_split(
            net_args,
            training_args,
            device=device
        )
    else:
        run_exp(
            net_args,
            training_args,
            device
        )


if __name__ == '__main__':
    main()
