"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


# Log images
def log_input_image(x, opts):
    return tensor2im(x)


# Clip image to range(0,1)
def clip_img(x):
    img_tmp = x.clone()[0]
    img_tmp = (img_tmp + 1) / 2
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]


def clip(var):
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var


def transpose_clip_image(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    return clip(var)


def tensor2im(var):
    # var shape: (3, H, W)
    img = transpose_clip_image(var)
    return Image.fromarray(img.astype('uint8'))


def img_to_tensor(x):
    out = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])(x)
    return out


def downscale(x, scale_times=1):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
    return x


def upscale(x, scale_times=1):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
    return x