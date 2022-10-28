# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

# revised by stma 2022.9.28
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append(".")
sys.path.append("..")
from PIL import Image
from torchvision import utils
from data_utils import *

from hyperstyle.models.stylegan2.model import Generator
from hyperstyle.models.encoders.psp import get_keys
from nets import *


class Trainer(nn.Module):

    def __init__(self, config, opts, attr_num, attr):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.accumulation_steps = 16
        self.config = config
        self.attr_num = attr_num
        self.attr = attr
        self.opts = opts
        print('attr num:', self.attr_num)

        # Networks

        #DOLL module
        self.DOLL = DOLL(img_size=256, style_dim=9216, max_conv_dim=512)
        if self.opts.extra_init:
            self.init_params()
        # Latent Classifier
        self.Latent_Classifier = LCNet([9216, 2048, 512, 40],
                                       activ='leakyrelu')
        # StyleGAN Model

        resolution = self.config['resolution']
        self.StyleGAN = Generator(resolution, 512, 8)

        self.label_file = opts.label_file
        self.corr_ma = None

        # Optimizers
        self.params = list(self.DOLL.parameters())
        self.optimizer = torch.optim.Adam(self.params,
                                          lr=config['lr'],
                                          betas=(config['beta_1'],
                                                 config['beta_2']),
                                          weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['step_size'],
            gamma=config['gamma'])

    def init_params(self):
        for param in self.DOLL.parameters():
            if isinstance(param, nn.Conv2d):
                nn.init.xavier_uniform_(param.weight.data)
                nn.init.constant_(param.bias.data, 0.1)
            elif isinstance(param, nn.BatchNorm2d):
                param.weight.data.fill_(1)
                param.bias.data.zero_()
            elif isinstance(param, nn.Linear):
                param.weight.data.normal_(0, 0.01)
                param.bias.data.zero_()

    def initialize(self, stylegan_model_path, classifier_model_path):
        state_dict = torch.load(stylegan_model_path, map_location='cpu')
        # style transformer
        self.StyleGAN.load_state_dict(get_keys(state_dict, 'decoder'),
                                      strict=True)
        self.StyleGAN.eval()
        self.Latent_Classifier.load_state_dict(
            torch.load(classifier_model_path), strict=False)
        self.Latent_Classifier.eval()

    def L1loss(self, input, target):
        return nn.L1Loss()(input, target)

    def MSEloss(self, input, target):
        if isinstance(input, list):
            return sum(
                [nn.MSELoss()(input[i], target[i])
                 for i in range(len(input))]) / len(input)
        else:
            return nn.MSELoss()(input, target)

    def SmoothL1loss(self, input, target):
        return nn.SmoothL1Loss()(input, target)

    def CEloss(self, x, target, reduction='mean'):
        return nn.CrossEntropyLoss(reduction=reduction)(x, target)

    def BCEloss(self, x, target, reduction='mean'):
        return nn.BCEWithLogitsLoss(reduction=reduction)(x, target)

    def GAN_loss(self, x, real=True):
        if real:
            target = torch.ones(x.size()).type_as(x)
        else:
            target = torch.zeros(x.size()).type_as(x)
        return nn.MSELoss()(x, target)

    def ort_loss(self, w, w_unrelated, w_related, w_related_transform):
        loss_1 = torch.mm(w_unrelated, w_related.transpose(0, 1))
        loss_1 = torch.abs(loss_1)
        loss_2 = torch.mm(w_unrelated, w_related_transform.transpose(0, 1))
        loss_2 = torch.abs(loss_2)

        return loss_1.mean() + loss_2.mean()

    def get_correlation(self, attr_num, threshold=1):
        if self.corr_ma is None:
            labels = np.load(self.label_file)
            self.corr_ma = np.corrcoef(labels.transpose())
            self.corr_ma[np.isnan(self.corr_ma)] = 0
        corr_vec = np.abs(self.corr_ma[attr_num:attr_num + 1])
        corr_vec[corr_vec >= threshold] = 1
        return 1 - corr_vec

    def get_coeff(self, x):

        sign_0 = F.relu(x - 0.5).sign()
        sign_1 = F.relu(0.5 - x).sign()
        return sign_0 * (-x) + sign_1 * (1 - x)

    def compute_loss(self, w, mask_input, n_iter):
        self.w_0 = w
        predict_label_logits_0 = self.Latent_Classifier(
            self.w_0.view(w.size(0), -1))
        label_0 = torch.sigmoid(predict_label_logits_0)
        attr_prob_0 = label_0[:, self.attr_num]
        # Get scaling factor
        coeff = self.get_coeff(attr_prob_0)
        target_prob = torch.clamp(attr_prob_0 + coeff, 0, 1).round()
        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_prob.type_as(attr_prob_0) - 1

        # Apply latent transformation
        _, w_unrelated, w_related, w_related_transform = self.DOLL(
            self.w_0.view(w.size(0), -1))

        coeff = coeff.reshape(w.size(0), -1)
        self.w_1 = w_unrelated + w_related + coeff * (w_related_transform -
                                                      w_related)

        self.w_1 = self.w_1.view(w.size())
        predict_label_logits_1 = self.Latent_Classifier(
            self.w_1.view(w.size(0), -1))

        # re loss
        T_coeff = target_prob.size(0) / (target_prob.sum(0) + 1e-8)
        F_coeff = target_prob.size(0) / (target_prob.size(0) -
                                         target_prob.sum(0) + 1e-8)
        mask_prob = T_coeff.float() * target_prob + F_coeff.float() * (
            1 - target_prob)
        self.loss_re = self.BCEloss(predict_label_logits_1[:, self.attr_num],
                                    target_prob,
                                    reduction='none') * mask_prob
        self.loss_re = self.loss_re.mean()

        # un loss
        threshold_val = 1 if 'corr_threshold' not in self.config else self.config[
            'corr_threshold']
        mask = torch.tensor(
            self.get_correlation(
                self.attr_num,
                threshold=threshold_val)).type_as(predict_label_logits_0)
        mask = mask.repeat(predict_label_logits_0.size(0), 1)
        self.loss_un = self.MSEloss(predict_label_logits_1 * mask,
                                    predict_label_logits_0 * mask)

        lambda_reg, lambda_re, lambda_un, lambda_ort = self.config['lambda']['reg'], self.config['lambda']['re'], \
                                                         self.config['lambda']['un'], self.config['lambda']['ort']

        self.loss = lambda_re * self.loss_re + lambda_un * self.loss_un
        # ort loss
        if lambda_ort > 0:
            self.loss_ort = self.ort_loss(w, w_unrelated, w_related,
                                          w_related_transform)
            self.loss += lambda_ort * self.loss_ort
        # Latent reg loss
        if lambda_reg > 0:
            self.loss_reg = self.MSEloss(self.w_1, self.w_0)
            self.loss += lambda_reg * self.loss_reg
        # identity loss: to be add

        # total loss
        return self.loss

    def get_image_training(self, w, w_1, weights_deltas):
        with torch.no_grad():
            x_0, _ = self.StyleGAN([w],
                                   input_is_latent=True,
                                   randomize_noise=False,
                                   weights_deltas=weights_deltas)
            x_1, _ = self.StyleGAN([w_1],
                                   input_is_latent=True,
                                   randomize_noise=False,
                                   weights_deltas=weights_deltas)
        return x_0, x_1

    def get_image(self, w, weights_deltas):
        with torch.no_grad():
            # Original image
            predict_label_logits_0 = self.Latent_Classifier(
                w.view(w.size(0), -1))
            label_0 = torch.sigmoid(predict_label_logits_0)
            attr_prob_0 = label_0[:, self.attr_num]
            print('get image,origin label', attr_prob_0)
            coeff = self.get_coeff(attr_prob_0)
            target_prob = torch.clamp(attr_prob_0 + coeff, 0, 1).round()
            if 'alpha' in self.config and not self.config['alpha']:
                coeff = 2 * target_prob.type_as(attr_prob_0) - 1
            print('get image,origin label:', attr_prob_0, 'coeff:', coeff)
            #coeff = 5 * coeff

            _, w_unrelated, w_related, w_related_transform = self.DOLL(
                w.view(w.size(0), -1))
            w_1 = w_unrelated + w_related + coeff * (w_related_transform -
                                                     w_related)

            w_1 = w_1.view(w.size())

            self.x_0, _ = self.StyleGAN([w],
                                        input_is_latent=True,
                                        randomize_noise=False,
                                        weights_deltas=weights_deltas)
            self.x_1, _ = self.StyleGAN([w_1],
                                        input_is_latent=True,
                                        randomize_noise=False,
                                        weights_deltas=weights_deltas)

    def log_image(self, w, n_iter, weights_deltas):
        with torch.no_grad():
            self.get_image(w, weights_deltas)
        img_dir = os.path.join(self.opts.log_path, 'image_' + self.attr)
        os.makedirs(img_dir, exist_ok=True)
        input_name = os.path.join(img_dir,
                                  'iter' + str(n_iter + 1) + '_input.jpg')

        modify_name = os.path.join(img_dir,
                                   'iter' + str(n_iter + 1) + '_modify.jpg')
        input_img = transpose_clip_image(downscale(self.x_0,
                                                   2)[0]).astype('uint8')

        modify_img = transpose_clip_image(downscale(self.x_1,
                                                    2)[0]).astype('uint8')
        Image.fromarray(input_img).convert('RGB').save(input_name)
        Image.fromarray(modify_img).convert('RGB').save(modify_name)

    def log_loss(self, n_iter):
        print('ort loss:', self.loss_ort.item(), 're loss:',
              self.loss_re.item(), ',un loss:', self.loss_un.item(),
              ',reg loss:', self.loss_reg.item(), ',total loss:',
              self.loss.item())

    def save_image(self, log_dir, n_iter):
        utils.save_image(clip_img(self.x_0),
                         log_dir + 'iter' + str(n_iter + 1) + '_img.jpg')
        utils.save_image(
            clip_img(self.x_1),
            log_dir + 'iter' + str(n_iter + 1) + '_img_modify.jpg')

    def save_model(self, log_dir):
        torch.save(self.DOLL.state_dict(),
                   log_dir + '/DOLL_' + str(self.attr) + '.pth.tar')

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'DOLL_state_dict': self.DOLL.state_dict(),
            'opt_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        if (n_epoch + 1) % 10 == 0:
            torch.save(
                checkpoint_state,
                '{:s}/checkpoint'.format(log_dir) + '_' + str(n_epoch + 1))
        else:
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.DOLL.load_state_dict(state_dict['DOLL_state_dict'])
        self.optimizer.load_state_dict(state_dict['opt_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, w, mask, n_iter):
        self.n_iter = n_iter
        self.optimizer.zero_grad()
        self.compute_loss(w, mask, n_iter).backward()
        self.optimizer.step()
