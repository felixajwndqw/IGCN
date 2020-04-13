import argparse
import math


def parse_none(x):
    return None if x == 'None' else x


class ExperimentParser(argparse.ArgumentParser):
    """Constructs an argument parser that returns arguments in groups

    Arguments are divided into network and training arguments. This is just a
    LITTLE bit hacky :)
    """
    def __init__(self, description=''):
        super().__init__(description=description)
        self.n_parser = self.add_argument_group('net')
        self.t_parser = self.add_argument_group('training')
        self.construct_parsers()

    def construct_parsers(self):
        self.n_parser.add_argument(
            '--dataset',
            default='mnistrot', type=str,
            choices=['mnist', 'mnistrotated', 'mnistrot', 'mnistrp', 'cifar', 'isbi', 'bsd', 'cirrus'],
            help='Type of dataset. Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--kernel_size',
            default=3, type=int,
            help='Kernel size')
        self.n_parser.add_argument(
            '--base_channels',
            default=16, type=int,
            help='Number of feature channels in first layer of network.')
        self.n_parser.add_argument(
            '--no_g',
            default=4, type=int,
            help='Number of Gabor filters.')
        self.n_parser.add_argument(
            '--dropout',
            default=0.3, type=float,
            help='Probability of dropout layer(s).')
        self.n_parser.add_argument(
            '--inter_gp',
            default=None, type=parse_none,
            choices=[None, 'max', 'mag', 'avg', 'sum'],
            help='Type of pooling to apply across Gabor '
                 'axis for intermediate layers. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--final_gp',
            default=None, type=parse_none,
            choices=[None, 'max', 'mag', 'avg', 'sum'],
            help='Type of pooling to apply across Gabor axis for the final layer. '
                 '(default: %(default)s)')
        self.n_parser.add_argument(
            '--all_gp',
            default=False, action='store_true',
            help='Whether to apply Gabor pooling on all layers')
        self.n_parser.add_argument(
            '--relu_type',
            default='c', type=str,
            choices=['c', 'mod', 'z'],
            help='Type of relu layer. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--fc_type',
            default='cat', type=str,
            choices=['cat', 'mag'],
            help='How complex tensors are combined into real tensors prior to FC. '
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--fc_block',
            default='lin', type=str,
            choices=['lin', 'clin', 'cnv'],
            help='Linear, complex linear or 1x1 conv for FC layers. Former will not be applied '
                 'before projection.'
                 'Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--cmplx',
            default=False, action='store_true',
            help='Whether to use a complex architecture.')
        self.n_parser.add_argument(
            '--single',
            default=False, action='store_true',
            help='Whether to use a single gconv layer between each pooling layer.')
        self.n_parser.add_argument(
            '--pooling',
            default='mag', type=str,
            choices=['max', 'mag', 'avg'],
            help='Type of pooling. Choices: %(choices)s (default: %(default)s)')
        self.n_parser.add_argument(
            '--nfc',
            default=2, type=int,
            help='Number of fully connected layers before classification.')
        self.n_parser.add_argument(
            '--bnorm',
            default='new', type=str,
            choices=['new', 'old'],
            help='New = proper bnorm /w momentum. Old = plain normalisation.')
        self.n_parser.add_argument(
            '--weight_init',
            default=None,
            type=parse_none,
            choices=[None, 'he', 'glorot'],
            help=('Type of weight initialisation. Choices: %(choices)s '
                  '(default: %(default)s, corresponding to re/im independent He init.)'))

        self.t_parser.add_argument(
            '--epochs',
            default=250, type=int,
            help='Number of epochs to train over.')
        self.t_parser.add_argument(
            '--augment',
            default=False, action='store_true',
            help='Whether to apply data augmentation.')
        self.t_parser.add_argument(
            '--lr',
            default=1e-4, type=float,
            help='Learning rate.')
        self.t_parser.add_argument(
            '--lr_decay',
            default=1, type=float,
            help='Learning rate decay.')
        self.t_parser.add_argument(
            '--weight_decay',
            default=1e-7, type=float,
            help='Weight decay/l2 reg.')
        self.t_parser.add_argument(
            '--nsplits',
            default=1, type=int,
            help='Number of validation splits to train over')
        self.t_parser.add_argument(
            '--batch_size',
            default=32, type=int,
            help='Number of samples in each batch')

    def parse_group_args(self):
        args = self.parse_args()
        arg_groups = {}

        for group in self._action_groups:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            arg_groups[group.title] = argparse.Namespace(**group_dict)

        return arg_groups['net'], arg_groups['training']

    @staticmethod
    def args_to_str(args):
        kwargs = vars(args)
        return '_'.join([f'{key}={val}' for key, val in kwargs.items()])


def calculate_error(items):
    N = len(items)
    mean_items = sum(items) / N
    diff_sq_sum = sum((item - mean_items) ** 2 for item in items)
    return math.sqrt(diff_sq_sum / (N * (N - 1)))


def main():
    parser = ExperimentParser()
    net_args, training_args = parser.parse_group_args()
    print("Net args")
    print(parser.args_to_str(net_args))
    print("Training args")
    print(parser.args_to_str(training_args))


if __name__ == '__main__':
    main()
