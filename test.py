import os
import numpy as np
import torch
import glob
import argparse

from torch.autograd import grad
from torchvision import utils
from hyperstyle.models.stylegan2.model import Generator
from hyperstyle.models.encoders.psp import get_keys
from data_utils import *
from nets import *
from tqdm import tqdm
from configs.path_config import model_paths, ckpt_paths, data_paths


def main(opts):
    with torch.no_grad():
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

        edit_couple_path = os.path.join(opts.save_image_path, 'edit_couple')
        os.makedirs(edit_couple_path, exist_ok=True)
        print("out path:", edit_couple_path)

        DOLLnet = DOLL(style_dim=9216).to(device)
        DOLL_model_path = model_paths[opts.attribute]
        state_dict = torch.load(DOLL_model_path)
        DOLLnet.load_state_dict(state_dict)

        generator = Generator(1024, 512, 8).to(device)
        generator_state_dict = torch.load(opts.stylegan_model_path,
                                          map_location='cpu')
        generator.load_state_dict(get_keys(generator_state_dict, 'decoder'),
                                  strict=True)

        test_latents_list = [
            glob.glob1(opts.test_latent_path, ext) for ext in ['*pt']
        ]
        test_latents_list = [
            item for sublist in test_latents_list for item in sublist
        ]
        test_latents_list.sort()
        print("test_latents_list length:", len(test_latents_list))

        weights_list = [
            glob.glob1(opts.test_weights_delta_path, ext) for ext in ['*npy']
        ]
        weights_list = [item for sublist in weights_list for item in sublist]
        weights_list.sort()

        for idx in tqdm(range(len(test_latents_list))):
            latent_name = os.path.join(opts.test_latent_path,
                                       test_latents_list[idx])
            postfix = os.path.splitext(latent_name)[1]
            if postfix == '.npy':
                w = torch.from_numpy(np.load(latent_name),
                                     allow_pickle=True).to(device)
            elif postfix == '.pt':
                w = torch.tensor(torch.load(latent_name)).to(device)
            w = w.unsqueeze(0)
            weights_deltas = np.load(os.path.join(opts.test_weights_delta_path,
                                                  weights_list[idx]),
                                     allow_pickle=True)
            sample_deltas = [
                d if d is not None else None for d in weights_deltas
            ]
            _, w_unrelated, w_related, w_related_transform = DOLLnet(
                w.view(w.size(0), -1))

            if not opts.no_origin:
                x_0, _ = generator([w],
                                   input_is_latent=True,
                                   randomize_noise=False,
                                   weights_deltas=weights_deltas)
                res = x_0.data
            for coeff in np.arange(opts.coeff_min, opts.coeff_max, opts.step):
                #print(coeff)
                w_1 = w_unrelated + w_related + coeff * (w_related_transform -
                                                         w_related)
                w_1 = w_1.view(w.size())
                w_1 = torch.cat((w_1[:, :11, :], w[:, 11:, :]), dim=1)
                x_1, _ = generator([w_1],
                                   input_is_latent=True,
                                   randomize_noise=False,
                                   weights_deltas=sample_deltas)
                res = torch.cat([res, x_1.data],
                                dim=3) if not opts.no_origin else x_1.data
            img_name = os.path.splitext(test_latents_list[idx])[0]
            img_name = img_name.split('_')[-1]
            utils.save_image(clip_img(res),
                             edit_couple_path + '/%06d.jpg' % int(img_name))

        print('Inversion with image input complete!')
        return


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_latent_path',
                        type=str,
                        default=data_paths['test_latent'],
                        help='test dataset path')
    parser.add_argument('--test_weights_delta_path',
                        type=str,
                        default=data_paths['test_weights_delta'],
                        help='test weights delta path')
    parser.add_argument('--stylegan_model_path',
                        type=str,
                        default=ckpt_paths['hyperstyle'],
                        help='stylegan model path')
    parser.add_argument('--save_image_path',
                        type=str,
                        default='./test_data/',
                        help='validate save image path')
    parser.add_argument('--attribute',
                        type=str,
                        default='Eyeglasses',
                        choices=['Eyeglasses', 'Smiling', 'Gender', 'Age'],
                        help='Attribute to modify')
    parser.add_argument('--coeff_min',
                        type=float,
                        default=1,
                        help='coeff range for editing')
    parser.add_argument('--coeff_max',
                        type=float,
                        default=5,
                        help='coeff range for editing')
    parser.add_argument('--step',
                        type=float,
                        default=1,
                        help='coeff range step for editing')
    parser.add_argument('--no_origin',
                        action='store_true',
                        help='decide if only output editing')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='use multiple gpus')
    opts = parser.parse_args()

    main(opts)
