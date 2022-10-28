import torch
import torch.nn as nn
import sys
import glob
import numpy as np
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("..")
from lattrans_hyperstyle.nets import F_mapping


class DOLL(nn.Module):

    def __init__(self, img_size=256, style_dim=9216, max_conv_dim=512):
        super(DOLL, self).__init__()
        self.decomposer = nn.Sequential(nn.Linear(style_dim, style_dim),
                                        nn.ReLU(),
                                        nn.Linear(style_dim, style_dim),
                                        nn.ReLU(),
                                        nn.Linear(style_dim, style_dim))
        self.transformer = nn.Sequential(nn.Linear(style_dim, style_dim),
                                         nn.ReLU(),
                                         nn.Linear(style_dim, style_dim),
                                         nn.ReLU(),
                                         nn.Linear(style_dim, style_dim))

    def forward(self, latent):
        z_related = self.decomposer(latent)
        z_unrelated = latent - z_related
        z_related_transform = self.transformer(z_related)
        return latent, z_unrelated, z_related, z_related_transform


class DOLLResidual(nn.Module):

    def __init__(self, img_size=256, style_dim=9216, max_conv_dim=512):
        super(DOLLResidual, self).__init__()
        self.decomposer = nn.Sequential(nn.Linear(style_dim, style_dim),
                                        nn.ReLU(),
                                        nn.Linear(style_dim, style_dim),
                                        nn.ReLU(),
                                        nn.Linear(style_dim, style_dim))
        self.transformer = nn.Sequential(nn.Linear(style_dim, style_dim),
                                         nn.ReLU(),
                                         nn.Linear(style_dim, style_dim),
                                         nn.ReLU(),
                                         nn.Linear(style_dim, style_dim))

    def forward(self, latent):
        z_related = self.decomposer(latent)
        z_unrelated = latent - z_related
        z_delta = self.transformer(z_related)
        return latent, z_unrelated, z_related, z_delta
