import torch.nn as nn
import sys


class DOLL(nn.Module):

    def __init__(self, style_dim=9216):
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

    def __init__(self, style_dim=9216):
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


class LCNet(nn.Module):

    def __init__(self, fmaps=[9216, 2048, 512, 40], activ='relu'):
        super().__init__()
        # Linear layers
        self.fcs = nn.ModuleList()
        for i in range(len(fmaps) - 1):
            in_channel = fmaps[i]
            out_channel = fmaps[i + 1]
            self.fcs.append(nn.Linear(in_channel, out_channel, bias=True))
        # Activation
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'leakyrelu':
            self.activ = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.activ(layer(x))
        x = self.fcs[-1](x)
        return x
