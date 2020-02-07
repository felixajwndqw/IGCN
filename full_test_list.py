import math
import time
import torch
import torch.optim as optim
from quicktorch.utils import train, imshow
from quicktorch.data import mnist, cifar
from igcn import IGCN


def main():
    dsets = ['mnist']  # , 'cifar']
    names = [
             '3c', '5c', '7c', '9c', 'lpc',
             '3', '5', '7', '9', 'lp',
             ]  # '3', 
    no_gabors = [2, 4, 8, 16, 32]
    mgs = [(False, True),
           (True, False),
           (True, True)]
    # inter_mgs = [False, True]
    # final_mgs = [True, False]
    no_epochs = 1
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
            '\t' + "{:4.2f}".format(m['accuracy']) +
            "\t" + "{:4.2f}".format(m['precision']) +
            "\t" + "{:4.2f}".format(m['recall']) +
            "\t" + str(m['epoch']) +
            "\t\t" + str(no_epochs) +
            '\t\t' + "{:4.2f}".format(total_params) +
            '\t' + "{:3d}m{:2d}s".format(mins, secs))
    f.close()


def run_exp(dset, model_name, no_g, rot_pool, inter_mg, final_mg, no_epochs, device):
    print("Training igcn{} on {} with rot_pool={}, no_g={}, "
          "inter_mg={}, final_mg={}".format(model_name, dset, rot_pool,
                                            no_g, inter_mg, final_mg))
    cmplx = model_name[-1] == 'c'  # just don't name any models with last letter c lol

    if dset == 'mnist':
        if inter_mg:
            b_size = int(4096 // no_g)
        else:
            b_size = 4096
        if cmplx:
            b_size //= 2
        train_loader, test_loader, _ = mnist(batch_size=b_size, rotate=True,
                                             num_workers=8)
    if dset == 'cifar':
        train_loader, test_loader, _ = cifar(batch_size=2048)

    model = IGCN(no_g=no_g, model_name=model_name, dset=dset,
                 rot_pool=rot_pool, inter_mg=inter_mg,
                 final_mg=final_mg, cmplx=cmplx).to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    total_params = total_params/1000000
    print("Total # parameter: " + str(total_params) + "M")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    start = time.time()
    m = train(model, [train_loader, test_loader], save_best=True,
              epochs=no_epochs, opt=optimizer, device=device,
              sch=scheduler)

    time_taken = time.time() - start
    mins = int(time_taken // 60)
    secs = int(time_taken % 60)
    write_results(dset, model_name, no_g,
                  m, no_epochs,
                  total_params, mins, secs, rot_pool=rot_pool,
                  inter_mg=inter_mg, final_mg=final_mg, cmplx=cmplx)

    del(model)
    torch.cuda.empty_cache()

    return m


if __name__ == "__main__":
    main()
