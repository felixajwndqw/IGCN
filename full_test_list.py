import torch
import torch.optim as optim
from quicktorch.utils import train, imshow
from quicktorch.data import mnist, cifar
from igcn import IGCN
import math


def main():
    dsets = ['mnist']  # , 'cifar']
    names = ['3', '5', '7', '9', 'lp']  # '3', 
    no_gabors = [2, 4, 8, 16, 32]
    inter_mgs = [False, True]
    final_mgs = [True, False]
    no_epochs = 100
    rot_pools = [True, False]
    accs = []
    epochs = []
    models = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for dset in dsets:
        for model_name in names:
            for no_g in no_gabors:
                for rot_pool in rot_pools:
                    for inter_mg in inter_mgs:
                        for final_mg in final_mgs:
                            a, e = run_exp(dset, model_name, no_g, rot_pool, inter_mg, final_mg, no_epochs, device)
                            accs.append(a)
                            epochs.append(e)
                            models.append(dset + "_" + model_name)

    print(accs, epochs, models)


def run_exp(dset, model_name, no_g, rot_pool, inter_mg, final_mg, no_epochs, device):
    print("Training igcn{} on {} with rot_pool={}, no_g={}, inter_mg={}, final_mg={}".format(model_name, dset, rot_pool, no_g, inter_mg, final_mg))
    if dset == 'mnist':
        if inter_mg:
            b_size = int(4096 // no_g)
        else:
            b_size = 4096
        train_loader, test_loader, _ = mnist(batch_size=b_size, rotate=True, num_workers=8)
    if dset == 'cifar':
        train_loader, test_loader, _ = cifar(batch_size=2048)

    model = IGCN(no_g=no_g, model_name=model_name, dset=dset, rot_pool=rot_pool, inter_mg=inter_mg, final_mg=final_mg).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_mem = total_params*32/1000000
    print("Total parameter size: " + str(total_mem) + "M")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    a, e = train(model, [train_loader, test_loader], save=False,
                 epochs=no_epochs, opt=optimizer, device=device)

    f = open("results.txt", "a+")
    f.write("\n" + dset + "\t" + model_name + "\t\t" + str(no_g)
            + "\t\t" + str(rot_pool) + "\t" + str(inter_mg) + "\t"
            + str(final_mg) + '\t' + str(a) + "\t" + str(e)
            + "\t\t" + str(no_epochs) + '\t\t'
            + str(total_mem))
    f.close()
    del(model)
    torch.cuda.empty_cache()

    return a, e


if __name__ == "__main__":
    main()