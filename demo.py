import torch
import torch.optim as optim
from quicktorch.utils import train, imshow
from quicktorch.data import mnist, cifar
from igcn import IGCN


dset = 'mnist'
if dset == 'mnist':
    train_loader, test_loader, _ = mnist(batch_size=512, rotate=True)
if dset == 'cifar':
    train_loader, test_loader, _ = cifar(batch_size=512)


example = iter(train_loader).next()
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
model = IGCN(no_g=4, model_name="7", dset=dset).cuda()
test_out = model(example[0].cuda())
for params in model.parameters():
    print(params.size())

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameter size: " + str(total_params*32/1000000) + "M")

optimizer = optim.SGD(model.parameters(), lr=0.1)
train(model, [train_loader, test_loader], save=False, epochs=5, opt=optimizer)

example = iter(test_loader).next()
test_out = model(example[0].cuda())
imshow(example[0], test_out, classes)
