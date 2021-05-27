from segmentation import run_seg_exp
import numpy as np
import torch
import pickle as pkl

from skopt import gp_minimize

from utils import ExperimentParser
from classify import run_classification_split, run_exp
from quicktorch.writers import LabScribeWriter


def dump(res):
    with open('results.pkl', 'wb') as fp:
        pkl.dump(res.x, fp)


def optimize(n_trials):
    """Run the gp_minimize function"""
    opt_bounds = [
        # (.000001, .1),   # weight decay
        # (.0, .8),   # dropout
        # (1e-6, 1e-1),   # learning rate
        # (0.6, 1),       # lr decay
        # (0, 3),        # no gabors
        (0, 1),         # pooling
        (0, 1),         # final_gp
        (0, 1),         # relu_type
        # (0, 1),         # modulation
        # (0, 1),         # batch norm
        # (0, 1),         # fc block
        # (0, 1),         # model type
    ]

    x0 = [0, 0, 0]

    print(gp_minimize(
        wrapper_function,
        opt_bounds,
        x0=x0,
        # y0=y0,
        n_calls=n_trials,
        verbose=True,
        callback=dump
    ))


def set_args(net_args, training_args, args):
    training_args.epochs = 30
    training_args.augment = True
    training_args.nsplits = 3
    training_args.batch_size = 2

    net_args.model_type = 'SFC'
    net_args.base_channels = 8
    net_args.dataset = 'isbi'

    # training_args.lr = args['lr']
    # training_args.lr_decay = args['lr_decay']
    # training_args.weight_decay = args['weight_decay']
    # net_args.dropout = args['dropout']

    # net_args.no_g = args['no_g']
    net_args.pooling = args['pooling']
    # net_args.inter_gp = args['gabor_pooling']
    net_args.final_gp = args['gabor_pooling']
    net_args.relu_type = args['relu_type']
    # net_args.mod = args['mod']
    # net_args.bnorm = args['bnorm']
    # net_args.fc_block = args['fc_type']
    # net_args.model_type = args['model_type']

    # training_args.lr = 0.00019
    # training_args.lr_decay = 0.9614
    # training_args.weight_decay = 0.001
    # net_args.dropout = .7
    # training_args.lr = 0.000114
    # training_args.lr_decay = 0.96
    # training_args.weight_decay = 0.0034
    # net_args.dropout = .3

    # default classification
    # net_args.no_g = 1
    # net_args.pooling = 'mag'
    # net_args.final_gp = 'avg'
    # net_args.relu_type = 'c'
    # net_args.mod = 'cmplx'
    # net_args.bnorm = 'new'
    # net_args.fc_block = 'lin'

    # default seg
    net_args.no_g = 1
    # net_args.pooling = 'max'
    # net_args.final_gp = 'max'
    # net_args.relu_type = 'c'
    net_args.mod = 'cmplx'
    net_args.bnorm = 'new'
    net_args.fc_block = 'lin'

    training_args.lr = 0.001
    training_args.lr_decay = 1.
    training_args.weight_decay = 1e-5
    net_args.dropout = .0


def wrapper_function(opt_args):
    parser = ExperimentParser(description='Handles MNIST/CIFAR tasks.')
    net_args, training_args = parser.parse_group_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    no_gs = (1, 2, 4, 8)
    pools = ('max', 'avg', 'mag')
    relus = ('c', 'mod')
    mods = ('hadam', 'cmplx')
    bnorms = ('old', 'new')
    fc_blocks = ('clin', 'lin')
    model_types = ('SFC', 'SFCNC', 'SFCResNet')

    args = {
        # 'weight_decay': opt_args[0],
        # 'dropout': opt_args[1],
        # 'lr': opt_args[2],
        # 'lr_decay': opt_args[3],
        # 'no_g': no_gs[opt_args[0]],
        'pooling': pools[opt_args[0]],
        'gabor_pooling': pools[opt_args[1]],
        'relu_type': relus[opt_args[2]],
        # 'mod': mods[opt_args[4]],
        # 'bnorm': bnorms[opt_args[5]],
        # 'fc_type': fc_blocks[opt_args[4]],
        # 'model_type': model_types[opt_args[5]]
    }

    set_args(net_args, training_args, args)

    print(vars(training_args))
    print(vars(net_args))
    print(args)

    # writer = LabScribeWriter(
    #     'Results',
    #     exp_name='SFC-'+'-'.join([f'{key}={val}' for key, val in args.items()]),
    #     exp_worksheet_name='MNIST',
    #     metrics_worksheet_name='MNISTMetrics',
    #     nsplits=1
    # )
    # writer.begin_experiment(args)

    # metrics, _ = run_classification_split(
    #     net_args,
    #     training_args,
    #     device=device,
    #     writer=writer
    # )

    # writer.upload_split({k: metrics[k] for k in ('accuracy', 'precision', 'recall')})

    # _, acc = run_exp(
    #     net_args,
    #     training_args,
    #     device=device,
    #     exp_name='SFC-'+'-'.join([f'{key}={val}' for key, val in args.items()])
    # )

    _, acc = run_seg_exp(
        net_args,
        training_args,
        device=device,
        args=TempArgs(
            model_variant='SFC',
            only_val=True
        ),
        exp_name='SFC-'+'-'.join([f'{key}={val}' for key, val in args.items()])
    )

    return 1 - acc


class TempArgs():
    def __init__(
        self,
        model_variant='SFC',
        model_path='',
        dir='../data/isbi',
        only_val=True,
        denoise=False,
        val_size=5
    ):
        super().__init__()
        self.model_variant = model_variant
        self.model_path = model_path
        self.dir = dir
        self.only_val = only_val
        self.denoise = denoise
        self.val_size = val_size


if __name__ == '__main__':
    optimize(15)
