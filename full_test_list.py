import torch
import torch.optim as optim
from quicktorch.utils import train, imshow
from quicktorch.data import mnist, cifar
from igcn import IGCN


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__, 
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                   type(obj.data).__name__, 
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "", 
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    print("Total size:", total_size)


def main():
    dsets = ['mnist']  # , 'cifar']
    names = ['3', '5', '7', '9']
    no_gabors = [4, 8, 16, 32]  # [2, 4, 8, 16, 32]
    max_gabor = [False, True]
    no_epochs = 50
    rot_pools = [True]
    accs = []
    epochs = []
    models = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    for dset in dsets:
        for model_name in names:
            for no_g in no_gabors:
                for rot_pool in rot_pools:
                    for max_g in max_gabor:
                        print("Training igcn{} on {} with rot_pool={}, no_g={}, max_g={}".format(model_name, dset, rot_pool, no_g, max_g))
                        if dset == 'mnist':
                            train_loader, test_loader, _ = mnist(batch_size=2048, rotate=True, num_workers=8)
                        if dset == 'cifar':
                            train_loader, test_loader, _ = cifar(batch_size=2048)

                        model = IGCN(no_g=no_g, model_name=model_name, dset=dset, rot_pool=rot_pool, max_gabor=max_g).to(device)

                        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print("Total parameter size: " + str(total_params*32/1000000) + "M")

                        optimizer = optim.Adam(model.parameters())
                        a, e = train(model, [train_loader, test_loader], save=False,
                                    epochs=no_epochs, opt=optimizer, device=device)
                        accs.append(a)
                        epochs.append(e)
                        models.append(dset+"_"+model_name)
                        f = open("results.txt", "a+")
                        f.write("\n" + dset + "\t" + model_name + "\t\t" + str(no_g) + "\t\t" + str(rot_pool) + "\t" + str(max_g) + "\t" + str(a) + "\t" + str(e) + "\t\t" + str(no_epochs))
                        f.close()
                        del(model)
                        torch.cuda.empty_cache()
                        dump_tensors()

    print(accs, epochs, models)



if __name__ == "__main__":
    main()