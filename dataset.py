from tkinter import W
from PIL import Image
import torch.utils.data as data
import sys

sys.path.append(".")
sys.path.append("..")
from .data_utils import make_dataset
from torchvision.transforms import transforms
import glob
import os
import torch
import numpy as np


class ImageDataset(data.Dataset):

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


class LatentDataset(data.Dataset):

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


class TestDataset(data.Dataset):

    def __init__(
            self,
            latent_dir='/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024/latents.npy',
            weights_delta_dir='/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024_save_weight/weight_deltas/',
            is_hyperstyle=True):
        self.latent_dir = latent_dir
        self.is_hyperstyle = is_hyperstyle
        self.weights_delta_dir = weights_delta_dir
        print('load test latent from:', latent_dir)
        if is_hyperstyle:
            self.latents = np.load(
                self.latent_dir,
                allow_pickle=True).item()  #(image_name,latents) pair
            self.length = len(self.latents)
            print(type(self.latents))
        else:
            self.latents = torch.load(self.latent_dir)
            self.length = self.latents.shape[0]
        print("test_latents_list length:", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_hyperstyle:
            #print(self.latents[idx])
            image_idx = self.latents[idx][0].split(',')[0]
            weights_delta = np.load(os.path.join(self.weights_delta_dir,
                                                 image_idx + '.npy'),
                                    allow_pickle=True)
            return self.latents[idx][1], weights_delta
        return self.latents[idx]


class TestHyperstyleDataset(data.Dataset):

    def __init__(
            self,
            latent_dir='/mnt/pami23/stma/EXP/test_latents/hyperstyle/',
            weights_delta_dir='/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024_save_weight/weight_deltas/',
            is_hyperstyle=False):
        self.latent_dir = latent_dir
        self.weights_delta_dir = weights_delta_dir
        print('load test latent from:', latent_dir)
        latents_list = [glob.glob1(latent_dir, ext) for ext in ['*npy']]
        latents_list = [item for sublist in latents_list for item in sublist]
        latents_list.sort()

        self.latents_list = latents_list

        if is_hyperstyle:
            weights_list = [
                glob.glob1(weights_delta_dir, ext) for ext in ['*npy']
            ]
            weights_list = [
                item for sublist in weights_list for item in sublist
            ]
            weights_list.sort()
            self.weights_list = weights_list
        else:
            self.weights_list = None
        self.length = len(latents_list)
        print("test_latents_list length:", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latents_list[idx])
        latent = np.load(latent_path)
        if self.weights_list is not None:
            weights_path = os.path.join(self.weights_delta_dir,
                                        self.weights_list[idx])
            weights_deltas = np.load((weights_path), allow_pickle=True)
            print(type(latent), type(weights_deltas))
            return (latent, weights_deltas)
        else:
            return latent


if __name__ == '__main__':
    latent_dir = '/mnt/pami23/yfyuan/EXP/LATENT/pSp/celeba_hq_test/'
    latents_list = [glob.glob1(latent_dir, ext) for ext in ['*pt']]
    latents_list = [item for sublist in latents_list for item in sublist]
    latents_list.sort()
    out_path = '/mnt/pami23/stma/EXP/test_latents/e4e'
    os.makedirs(out_path, exist_ok=True)
    for latent_name in latents_list:
        temp_latent_path = os.path.join(latent_dir, latent_name)
        latent = torch.load(temp_latent_path)
        print(type(latent))
        latent = np.array(latent)
        img_name = os.path.splitext(latent_name)[0]
        img_name = img_name.split('_')[-1]
        print(img_name)
        #np.save(os.path.join(out_path, "%06d.npy" % int(img_name)), latent)
