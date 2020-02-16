import argparse
import time
import torch
import torch.optim as optim
from quicktorch.utils import train, evaluate
from quicktorch.data import mnist, cifar, mnistrot
from igcn import IGCN


def main():
    dsets = ['mnistrot']  # , 'cifar']
    names = [
            #  '3o', '5o', '7o', '9o',
             '3oc', '5oc',
             ]
    no_gabors = [2, 4, 8, 16]
    mgs = [
           (False, True),
           (True, False),
           (True, True)
    ]
    # inter_mgs = [False, True]
    # final_mgs = [True, False]
    no_epochs = 300
    rot_pools = [False]
    metrics = []
    models = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for dset in dsets:
        for model_name in names:
            for no_g in no_gabors:
                for rot_pool in rot_pools:
                    for inter_mg, final_mg in mgs:
                            m = run_exp(dset, model_name, no_g,
                                        rot_pool, inter_mg, final_mg,
                                        no_epochs, device)
                            metrics.append(m)
                            models.append(dset + "_" + model_name)

    print(metrics, models)


def write_results(dset, model_name, no_g, m, no_epochs,
                  total_params, mins, secs, rot_pool=False,
                  inter_mg=False, final_mg=False, cmplx=False):
    f = open("results.txt", "a+")
    f.write("\n" + dset +
            "\t" + model_name +
            "\t\t" + str(no_g) +
            "\t\t" + str(rot_pool) +
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


def run_exp(dset, model_name, no_g, rot_pool, inter_mg, final_mg, no_epochs, device):
    print("Training igcn{} on {} with rot_pool={}, no_g={}, "
          "inter_mg={}, final_mg={}".format(model_name, dset, rot_pool,
                                            no_g, inter_mg, final_mg))
    cmplx = 'c' in model_name
    one = 'o' in model_name

    if dset == 'mnist' or dset == 'mnistrot':
        if inter_mg or final_mg:
            b_size = int(4096 // no_g)
        else:
            b_size = 4096
        if cmplx:
            b_size //= 4
        if dset == 'mnist':
            train_loader, test_loader, _ = mnist(batch_size=b_size,
                                                 rotate=True,
                                                 num_workers=8)
        if dset == 'mnistrot':
            train_loader, test_loader, _ = mnistrot(batch_size=b_size,
                                                    num_workers=8)

    if dset == 'cifar':
        train_loader, test_loader, _ = cifar(batch_size=2048)

    model = IGCN(no_g=no_g, model_name=model_name, dset=dset,
                 rot_pool=rot_pool, inter_mg=inter_mg,
                 final_mg=final_mg, cmplx=cmplx, one=one).to(device)

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
    write_results(dset, model_name, no_g,
                  m, no_epochs,
                  total_params, mins, secs, rot_pool=rot_pool,
                  inter_mg=inter_mg, final_mg=final_mg, cmplx=cmplx)

    del(model)
    torch.cuda.empty_cache()

    return m


def test_num_workers(device):
    b_sizes = [512, 1024, 2048]
    n_workers = [8, 16, 32]
    model = IGCN(no_g=16, model_name='9c', dset='mnist',
                 rot_pool=False, inter_mg=True,
                 final_mg=True, cmplx=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for b_s in b_sizes:
        for n in n_workers:
            print(f'Starting training with b={b_s}, n={n}')
            train_loader, test_loader, _ = mnist(batch_size=b_s, rotate=True,
                                                 num_workers=n)

            m = train(model, [train_loader, test_loader], save_best=True,
                      epochs=1, opt=optimizer, device=device,
                      sch=scheduler)
            print(f'Finished training with b={b_s}, n={n}')
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handles MNIST/CIFAR tasks.')
    parser.add_argument('--full',
                        default=False, action='store_true',
                        help='Whether to run full test list. (default: %(default)s)')

    parser.add_argument('--dataset',
                        default='mnistrot', type=str,
                        choices=['mnist', 'mnistrotated', 'mnistrot', 'cifar'],
                        help='Type of dataset. Choices: %(choices)s (default: %(default)s)')
    parser.add_argument('--model_name',
                        default='3oc', type=str,
                        choices=['3o', '5o', '7o', '9o', '3oc', '5oc', '7oc', '9oc'],
                        help='Type of dataset. Choices: %(choices)s (default: %(default)s)')
    parser.add_argument('--no_g',
                        default=4, type=int,
                        help='Number of Gabor filters.')
    parser.add_argument('--inter_mg',
                        default=False, action='store_true',
                        help='Whether to pool over orientations in intermediate layers. (default: %(default)s)')
    parser.add_argument('--final_mg',
                        default=False, action='store_true',
                        help='Whether to pool over orientations in final layers. (default: %(default)s)')

    parser.add_argument('--epochs',
                        default=250, type=int,
                        help='Number of epochs to train over.')
    parser.add_argument('--batch_size',
                        default=8, type=int,
                        help='Number of samples in each batch.')
    args = parser.parse_args()

    if args.full:
        main()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        m = run_exp(args.dataset, args.model_name, args.no_g,
                    False, args.inter_mg, args.final_mg,
                    args.epochs, device)
    # test_num_workers(0)
