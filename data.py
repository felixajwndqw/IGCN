import glob
import os
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from quicktorch.datasets import ClassificationDataset
from quicktorch.utils import imshow


class CirrusDataset(Dataset):
    """Loads a classification dataset from file.

    Assumes labels are stored in a CSV file with the images in the same folder.
    It seems a little unintuitive and unnecessarily restrictive to support only
    passing a CSV filename for initialisation. Perhaps I will change this at
    some point.

    Args:
        csv_file (str, optional): Filename of csv file.
            Extension not necessary.
        transform (torchvision.transforms.Trasform, optional): Transform(s) to
        be applied to the data.
        **kwargs:
            weights_url (str, optional): A URL to download pre-trained weights.
            name (str, optional): See above. Defaults to None.
    """
    def __init__(self, img_dir,
                 transform=None, target_transform=None):
        self.image_paths = [img
                            for img in glob.glob(img_dir+'/*.png')]

        self.num_classes = 2
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image = transforms.ToTensor()(
            Image.open(self.image_paths[i]).convert('L')
        )
        cirrus = image[:, 80:400, :image.size(2)//2]
        label = image[:, 80:400, image.size(2)//2:]
        if self.transform is not None:
            cirrus = self.transform(cirrus)
        if self.target_transform is not None:
            label = self.target_transform(self.labels[i])
        # print(cirrus.size(), label.size())
        return cirrus, label

    def __len__(self):
        return len(self.image_paths)


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


if __name__ == '__main__':
    cirri = DataLoader(CirrusDataset('data/cirrus_examples/train'),
                       batch_size=1, shuffle=True)

    for example_img, example_mask in cirri:
        imshow(example_img)
