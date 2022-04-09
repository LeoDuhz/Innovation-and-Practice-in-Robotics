import torch
from torch.utils import data
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import utils


def default_loader(path):
    return Image.open(path).convert("RGB")

def default_file_list_reader(file_list):
    image_list = []
    with open(file_list, 'r') as rf:
        for line in rf.readlines():
            image_path = line.strip()
            image_list.append(image_path)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    # check if the file with filename is image or not
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    # create a list of image paths
    for dir_paths, dir_names, filenames in sorted(os.walk(dir)):
        for filename in filenames:
            if is_image_file(filename):
                path = os.path.join(dir_paths, filename)
                images.append(path)

    return images

class ImageFolder(data.Dataset):
    def __int__(self, root, transform=None, return_paths=None, loader=default_loader):

        images = make_dataset(root)
        if len(images):
            raise(RuntimeError("Found no image in: " + root + "\n"
                               "Support extensions are: " + ", ".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)



class CityscapeDataset(data.Dataset):
    def __init__(self, root, phase, transform_3=None):
        self.transform_3 = transform_3

        color_paths, image_paths = self.get_paths(root, phase)

        utils.natural_sort(color_paths)
        utils.natural_sort(image_paths)

        for path1, path2 in zip(color_paths, image_paths):
            assert self.paths_match(path1, path2), \
                "The label image pair (%s, %s) is not the right pair" % (path1, path2)

        self.color_paths = color_paths
        self.image_paths = image_paths

        size = len(self.color_paths)
        self.dataset_size = size


    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)

        return '_'.join(name1.split('_')[:3]) == '_'.join(name2.split('_')[:3])

    def get_paths(self, data_root, data_phase):
        root = data_root
        phase = 'val' if data_phase == 'test' else 'train'
        assert phase == 'val' or 'train'

        image_dir = os.path.join(root, 'leftImg8bit', phase)
        image_paths = make_dataset(image_dir)

        color_dir = os.path.join(root, 'gtFine', phase)
        color_path_all = make_dataset(color_dir)
        color_paths = [p for p in color_path_all if p.endswith('_color.png')]
        return color_paths, image_paths

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        image_path = self.image_paths[index]
        color = Image.open(color_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')

        color_tensor = self.transform_3(color)
        image_tensor = self.transform_3(image)

        color_tensor = color_tensor.float()
        image_tensor = image_tensor.float()

        input_dict = {'color': color_tensor,
                      'image': image_tensor,
                      'path': image_path}

        return input_dict

    def __len__(self):
        return self.dataset_size



