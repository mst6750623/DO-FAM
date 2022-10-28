from tkinter import W
from PIL import Image
from torch.utils.data import Dataset
import sys

sys.path.append(".")
sys.path.append("..")
from data_utils import make_dataset
from torchvision.transforms import transforms
import glob
import os
import torch
import numpy as np


class ImageDataset(Dataset):

    def __init__(self, image_dir, is_Train=True):

        imagesource = sorted(make_dataset(image_dir))
        train_len = (int)(0.9 * len(imagesource))

        # toTensor()可将0-255变成0-1之间
        self.from_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        # nomalize()可将0-1变成(0-0.5)/0.5=-1 ~ (1-0.5)/0.5=1之间
        self.to_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if is_Train:
            self.image = imagesource[:train_len]
        else:

            self.image = imagesource[train_len:]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # from是原图，to是增强后的图
        from_path = self.image[idx]
        from_image = Image.open(from_path).convert('RGB')
        to_image = self.to_transforms(from_image)
        from_image = self.from_transforms(from_image)

        return from_image, to_image


class LatentDataset(Dataset):

    def __init__(self, latent_dir, label_dir, training_set=True):
        postfix = os.path.splitext(latent_dir)[1]
        if postfix == '.npy':
            self.dlatents = np.load(latent_dir)
        elif postfix == '.pt':
            self.dlatents = torch.load(latent_dir)
        self.labels = np.load(label_dir)

        #无validate
        ''' train_len = int(0.9*len(labels))
        if training_set:
            self.dlatents = dlatents[:train_len] 
            self.labels = labels[:train_len]
            #self.process_score()
        else:
            self.dlatents = dlatents[train_len:]
            self.labels = labels[train_len:]'''

        self.length = self.dlatents.shape[0]
        print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dlatent = torch.tensor(self.dlatents[idx]).clone().detach()
        lbl = torch.tensor(self.labels[idx])

        return dlatent, lbl


class InferenceDataset(Dataset):

    def __init__(self, root, opts, transform=None):
        self.paths = sorted(make_dataset(root))
        self.transform = transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)
        return from_im