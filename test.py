import os
import numpy as np
import torch

from PIL import Image
from torch.autograd import grad
from torchvision import utils
import sys
import glob

sys.path.append(".")
sys.path.append("..")

from lattrans_hyperstyle.utils.functions import *
from lattrans_hyperstyle.nets import *
#from lattrans_hyperstyle.e4e.utils.common import *
import argparse

from lattrans_hyperstyle.hyperstyle.models.stylegan2.model import Generator
from lattrans_hyperstyle.pixel2style2pixel.models.psp import get_keys
from .my_l2mnet import L2MTransformer
from tqdm import tqdm


def main(opts):
    with torch.no_grad():
        device = 'cuda'
        l2m = L2MTransformer(img_size=256, style_dim=9216,
                             max_conv_dim=512).to(device)
        state_dict = torch.load(opts.l2m_model_path)
        '''state_dict = torch.load(
            '/home/stma/workspace/lattrans_hyperstyle/logs/l2m/l2m_Eyeglasses.pth.tar'
        )'''
        l2m.load_state_dict(state_dict)
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

        edit_couple_path = os.path.join(opts.save_image_path, 'edit_couple')
        os.makedirs(edit_couple_path, exist_ok=True)
        print("out path:", edit_couple_path)
        for idx in tqdm(range(len(test_latents_list))):

            w = torch.tensor(
                torch.load(
                    os.path.join(opts.test_latent_path,
                                 test_latents_list[idx]))).to(device)
            w = w.unsqueeze(0)
            weights_deltas = np.load(os.path.join(opts.test_weights_delta_path,
                                                  weights_list[idx]),
                                     allow_pickle=True)
            sample_deltas = [
                d if d is not None else None for d in weights_deltas
            ]
            _, w_unrelated, w_related, w_related_transform = l2m(
                w.view(w.size(0), -1))

            if not opts.no_origin:
                x_0, _ = generator([w],
                                   input_is_latent=True,
                                   randomize_noise=False,
                                   weights_deltas=weights_deltas)
                #to cpu to prevent the CUDA out of memory error
                x_0 = x_0.to('cpu')
                res = x_0
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
                x_1 = x_1.to('cpu')
                res = torch.cat([res, x_1.data],
                                dim=3) if not opts.no_origin else x_1.data
            img_name = os.path.splitext(test_latents_list[idx])[0]
            img_name = img_name.split('_')[-1]
            utils.save_image(clip_img(res),
                             edit_couple_path + '/%06d.jpg' % int(img_name))

        print('Inversion with image input complete!')
        return


if __name__ == '__main__':
    '''torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    Image.MAX_IMAGE_PIXELS = None'''
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
    parser.add_argument(
        '--test_latent_path',
        type=str,
        default='/mnt/pami23/yfyuan/EXP/LATENT/hyperstyle/celeba_hq_test/',
        help='test dataset path')
    parser.add_argument(
        '--test_weights_delta_path',
        type=str,
        default=
        '/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024_save_weight/weight_deltas/',
        help='test weights delta path')
    # parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
    parser.add_argument(
        '--label_file',
        type=str,
        default='/home/stma/workspace/VR-FAM/CelebAMask_anno_sorted.npy',
        help='label file path')
    parser.add_argument(
        '--stylegan_model_path',
        type=str,
        default=
        '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/hyperstyle_ffhq.pt',
        help='stylegan model path')
    parser.add_argument(
        '--l2m_model_path',
        type=str,
        default=
        '/home/stma/workspace/lattrans_hyperstyle/logs/l2m/l2m_Eyeglasses.pth.tar',
        help='stylegan model path')
    parser.add_argument(
        '--classifier_model_path',
        type=str,
        default=
        '/home/stma/workspace/yyf-latent-transformer/pretraining/checkpoint/001/latent_classifier_epoch_50.pth',
        help='pretrained attribute classifier')
    parser.add_argument(
        '--save_image_path',
        type=str,
        default=
        '/home/stma/workspace/lattrans_hyperstyle/l2m/inference/woortho/',
        help='validate save image path')
    parser.add_argument('--coeff_min',
                        type=float,
                        default=1.3,
                        help='coeff range for editing')
    parser.add_argument('--coeff_max',
                        type=float,
                        default=2,
                        help='coeff range for editing')
    parser.add_argument('--step',
                        type=float,
                        default=1,
                        help='coeff range step for editing')
    parser.add_argument('--no_origin',
                        action='store_true',
                        help='decide only output editing')

    parser.add_argument('--reverse',
                        action='store_true',
                        help='decide whether use reverse model or not')
    opts = parser.parse_args()

    main(opts)
