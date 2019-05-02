import torch
import torch.optim as optim
from quicktorch.utils import train, imshow
from quicktorch.data import mnist, cifar
from igcn import IGCN


dsets = ['mnist', 'cifar']
names = ['3', '5', '7', '9']
rot_pools = [False]
accs = []
epochs = []

for dset in dsets:
    for model_name in names:
        for rot_pool in rot_pools:
            print("Training igcn{} on {} with rot_pool={}".format(dset, model_name, rot_pool))
            if dset == 'mnist':
                train_loader, test_loader, _ = mnist(batch_size=512, rotate=True)
            if dset == 'cifar':
                train_loader, test_loader, _ = cifar(batch_size=512)

            model = IGCN(no_g=4, model_name="7", dset=dset, rot_pool=rot_pool).cuda()
            for params in model.parameters():
                print(params.size())

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total parameter size: " + str(total_params*32/1000000) + "M")

            optimizer = optim.SGD(model.parameters(), lr=0.1)
            a, e = train(model, [train_loader, test_loader], save=False, epochs=50, opt=optimizer)
            accs.append(a)
            epochs.append(e)

print(accs, epochs)