import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from igcn.seg.covariance.models import IGCNCovar, IGCNCovarTest, IGCNCovarTest2
from igcn.seg.covariance.loss import SegRegLoss
from igcn.seg.covariance.metrics import SegRegMetric
from quicktorch.utils import perform_pass
from data import SynthCirrusDataset, TensorList
from utils import ExperimentParser
from torchviz import make_dot
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            print("NONE")
            return False
        # print(grad_output.size())
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(
            style='filled',
            shape='box',
            align='left',
            fontsize='12',
            ranksep='0.1',
            height='0.2'
        )
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                # print(help(fn))
                print(fn)
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            print(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def register_fw_hooks(model):
    def fw_hook(m, i, o):
        print(m._get_name())
    for (mo) in model.modules():
        mo.register_forward_hook(fw_hook)


def collate_segreg(data):
    tensors = [
        torch.Tensor(len(data), *t.size()) for t in data[0]
    ]
    for i, d in enumerate(data):
        tensors[0][i] = d[0]
        tensors[1][i] = d[1]
        tensors[2][i] = d[2]
    out2 = TensorList((tensors[1], tensors[2]))
    return tensors[0], out2


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
    parser.add_argument('--dir',
                        default='../datasets/stars_bad_columns_ccd', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--denoise',
                        default=False, action='store_true',
                        help='Attempts to denoise the image')

    net_args, training_args = parser.parse_group_args()
    args = parser.parse_args()
    data_dir = args.dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    denoise = args.denoise
    split = [range(150), range(150, 200)]

    train_loader = DataLoader(
        SynthCirrusDataset(
            os.path.join(data_dir, 'train'),
            indices=split[0],
            denoise=denoise,
            angle=True),
        batch_size=4, shuffle=True,
        collate_fn=collate_segreg)

    dataset = os.path.split(data_dir)[-1]
    model = IGCNCovarTest2(
        name=f'igcn_dataset={dataset}_denoise={denoise}',
        n_channels=1,
        base_channels=2,
        no_g=4,
        n_classes=1,
        gp='max',
        angle_method=3,
        pooling='max'
    ).to(device)
    # register_fw_hooks(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=training_args.lr,
                           weight_decay=training_args.weight_decay)
    criterion = SegRegLoss()
    metrics = SegRegMetric()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, training_args.lr_decay)
    # m = train(model, [train_loader, val_loader], save_best=True,
    #           epochs=training_args.epochs, opt=optimizer, device=device,
    #           criterion=criterion,
    #           metrics=metrics,
    #           sch=scheduler)
    batch = next(iter(train_loader))
    model.train()
    mask, angle = model(batch[0].to(device))
    # loss = criterion((mask, angle), batch[1].to(device))
    seg_loss = torch.nn.MSELoss()(mask, batch[1][0].to(device))
    reg_loss = torch.nn.MSELoss()(angle, batch[1][1].to(device))
    loss = seg_loss + reg_loss

    get_dot = register_hooks(loss)
    loss.backward()
    # plot_grad_flow(model.named_parameters())
    # plt.show()

    dot = get_dot()
    dot.render()
    # make_dot(mask, params=dict(model.named_parameters()))



if __name__ == '__main__':
    main()
