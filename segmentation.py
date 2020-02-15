import sys
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from igcn.seg.models import UNetIGCN
from quicktorch.utils import train, imshow
from data import CirrusDataset, EMDataset


def main():
    parser = argparse.ArgumentParser(description='Runs a segmentation model')
    parser.add_argument('--dir',
                        default='data/halo1000', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--data',
                        default='cirrus', type=str,
                        choices=['cirrus', 'isbi'],
                        help='Type of dataset. Choices: %(choices)s (default: %(default)s)',
                        metavar='dataset')

    args = parser.parse_args()
    data_dir = args.dir

    Dataset = None
    if args.data == 'cirrus':
        Dataset = CirrusDataset
    elif args.data == 'isbi':
        Dataset = EMDataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = DataLoader(Dataset(os.path.join(data_dir, 'train')),
                            batch_size=32, shuffle=True)
    test_data = DataLoader(Dataset(os.path.join(data_dir, 'test')),
                           batch_size=32, shuffle=True)
    model = UNetIGCN(n_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # train(model, train_data, epochs=1, opt=optimizer, device=device)

    example = iter(test_data).next()
    print(example[0][0].size())
    test_out = model(example[0].to(device))
    imshow(torch.stack([example[0][0].cpu().detach(),
                        example[1][0].cpu().detach(),
                        torch.clamp(test_out[0], min=0, max=1).cpu().detach()]))


if __name__ == '__main__':
    main()
