import torch
import torch.optim as optim
from quicktorch.utils import train, imshow
from quicktorch.data import mnist, cifar
from igcn import IGCN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dset = 'mnist'
if dset == 'mnist':
    train_loader, test_loader, _ = mnist(batch_size=512, rotate=True)
if dset == 'cifar':
    train_loader, test_loader, _ = cifar(batch_size=512)


classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
model = IGCN(no_g=4, model_name="7", dset=dset).to(device)
for params in model.parameters():
    print(params.size())

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameter size: " + str(total_params/1000000) + "M")

optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
train(model, [train_loader, test_loader], epochs=1, opt=optimizer,
      device=device, sch=scheduler)

example = next(iter(test_loader))
test_out = model(example[0].to(device))
imshow(example[0], test_out, classes)
