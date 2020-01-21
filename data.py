from torchvision import transforms
from torch.utils.data import DataLoader
from quicktorch.datasets import ClassificationDataset


def gen_data(batch_size=3000):
    transform = transforms.Compose([
                transforms.RandomRotation(180),
                transforms.ToTensor(),
            ])
    test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    data = DataLoader(ClassificationDataset("data/textures/images/lbls.csv",
                      transform=transform),
                      batch_size=batch_size, shuffle=True)

    test_data = DataLoader(ClassificationDataset("data/textures/test_images/lbls.csv",
                           transform=test_transform),
                           batch_size=batch_size, shuffle=False)

    return data, test_data
