import matplotlib.pyplot as plt
import PIL.Image as Image

PATHS = [
    'data\cirrus300\constant\train\clean\clean_00000.png',
    'data\cirrus300\constant\train\input\cirrus_00000.png',
    'data\cirrus300\constant\train\target\mask_00000.png',
    'data\cirrus300\rotated\train\clean\clean_00000.png',
    'data\cirrus300\rotated\train\input\cirrus_00000.png',
    'data\cirrus300\rotated\train\target\mask_00000.png',
    'data\cirrus300\stars\train\clean\clean_00000.png',
    'data\cirrus300\stars\train\input\cirrus_00000.png',
    'data\cirrus300\stars\train\target\mask_00000.png',
]


def load_images(fpaths):
    images = []
    for fpath in fpaths:
        images.append(Image.open(fpath))
    return images


def main():
    images = load_images(PATHS)
    



if __name__ == "__main__":
    main()
