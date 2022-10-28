import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import argparse
import sys
import glob

sys.path.append(".")
sys.path.append("..")
from torchvision import utils
from lattrans_hyperstyle.utils.functions import clip_img
from hyperstyle.utils.common import tensor2im
from hyperstyle.configs.paths_config import edit_paths
from hyperstyle.models.stylegan2.model import Generator
from hyperstyle.models.encoders.psp import get_keys

torch.backends.cudnn.enabled = False

interfacegan_directions = {
    'male': torch.load(edit_paths['male']).cuda(),
    'age': torch.load(edit_paths['age']).cuda(),
    'smile': torch.load(edit_paths['smile']).cuda(),
    'pose': torch.load(edit_paths['pose']).cuda(),
    'eyeglasses': torch.load(edit_paths['eyeglasses']).cuda(),
}


def run(opts):

    device = 'cuda'
    stylegan_model_path = '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/hyperstyle_ffhq.pt'
    # update test options with options used during training
    #net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)
    generator = Generator(1024, 512, 8).to(device)
    generator_state_dict = torch.load(stylegan_model_path, map_location='cpu')
    generator.load_state_dict(get_keys(generator_state_dict, 'decoder'),
                              strict=True)
    generator.eval()
    print(opts.test_latent_path)
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

    edit_couple_path = os.path.join(opts.save_image_path, 'edit_results',
                                    opts.edit_attribute)
    os.makedirs(edit_couple_path, exist_ok=True)
    print("out path:", edit_couple_path)
    editing_direction = interfacegan_directions[
        opts.edit_attribute.lower()].to(device)
    '''editing_direction = np.load(
        '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/interfacegan_directions/stylegan_ffhq_eyeglasses_boundary.npy',
        allow_pickle=True)'''
    print("edit direction shape:", editing_direction.shape)
    for idx in tqdm(range(len(test_latents_list))):

        w = torch.load(
            os.path.join(opts.test_latent_path, test_latents_list[idx]))
        w = torch.tensor(w).to(device)
        w = w.unsqueeze(0)

        weights_deltas = np.load(os.path.join(opts.test_weights_delta_path,
                                              weights_list[idx]),
                                 allow_pickle=True)
        #sample_deltas = [d if d is not None else None for d in weights_deltas]
        if not opts.no_origin:
            x_0, _ = generator([w],
                               input_is_latent=True,
                               weights_deltas=weights_deltas,
                               randomize_noise=False)
            res = x_0.data
        for coeff in np.arange(opts.coeff_min, opts.coeff_max, opts.step):
            #print(coeff)
            w_1 = w + coeff * editing_direction
            w_1 = w_1.view(w.size())
            w_1 = torch.cat((w_1[:, :11, :], w[:, 11:, :]), dim=1)

            x_1, _ = generator([w_1],
                               input_is_latent=True,
                               weights_deltas=weights_deltas,
                               randomize_noise=False)
            res = torch.cat([res, x_1.data],
                            dim=3) if not opts.no_origin else x_1.data
        img_name = os.path.splitext(test_latents_list[idx])[0]
        img_name = img_name.split('_')[-1]
        utils.save_image(clip_img(res),
                         edit_couple_path + '/%06d.jpg' % int(img_name))

    print('Inversion with image input complete!')
    return


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    Image.MAX_IMAGE_PIXELS = None
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--edit_attribute',
                        type=str,
                        default='smile',
                        help='attribute')
    parser.add_argument(
        '--save_image_path',
        type=str,
        default=
        '/home/stma/workspace/lattrans_hyperstyle/l2m/inference/hyperstyle_edit/interface_editing_results/smile2.5',
        help='validate save image path')
    parser.add_argument('--coeff_min',
                        type=float,
                        default=2.5,
                        help='coeff range for editing')
    parser.add_argument('--coeff_max',
                        type=float,
                        default=3,
                        help='coeff range for editing')
    parser.add_argument('--step',
                        type=float,
                        default=1,
                        help='coeff range step for editing')
    parser.add_argument('--no_origin',
                        action='store_true',
                        help='decide only output editing')
    opts = parser.parse_args()

    run(opts)
