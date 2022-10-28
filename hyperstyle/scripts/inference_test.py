import os

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
from options.test_options import TestOptions
from models.stylegan2.model import Generator

device = torch.device('cuda')
def run_inversion_test(codes, opts, weights):
    decoder = Generator(opts.output_size, 512, 8, channel_multiplier=2)
    images, _ = decoder([codes], randomize_noise=False, input_is_latent=True, weights_delta=weights)

    return images




def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    # out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    # os.makedirs(out_path_coupled, exist_ok=True)
    #
    # os.makedirs(test_opts.latent_dir, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)

    print('Loading dataset for {}'.format(opts.dataset_type))
    # dataset_args = data_configs.DATASETS[opts.dataset_type]
    # transforms_dict = dataset_args['transforms'](opts).get_transforms()
    # dataset = InferenceDataset(root=opts.data_path,
    #                            transform=transforms_dict['transform_inference'],
    #                            opts=opts)
    # dataloader = DataLoader(dataset,
    #                         batch_size=opts.test_batch_size,
    #                         shuffle=False,
    #                         num_workers=int(opts.test_workers),
    #                         drop_last=False)
    #
    # if opts.n_images is None:
    #     opts.n_images = len(dataset)
    #



    global_time = []



    if "cars" in opts.dataset_type:
        resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
    else:
        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    with torch.no_grad():
        for latent_path in os.listdir(test_opts.latent_dir):
            if '.pt' in latent_path:
                fname, _ = os.path.splitext(latent_path)
                im_name = fname.split('_')[-1]
                # latent
                input_latent = torch.load(os.path.join(test_opts.latent_dir, latent_path))
                print(f'input: {input_latent.shape}')
                input_latent = input_latent[0]
                print(f'input2: {input_latent.shape}')
                input_latent = input_latent[np.newaxis,:]
                print(f'input3: {input_latent.shape}')
                input_latent = torch.Tensor(input_latent).to(device)
                # weights
                weights_delta = np.load(os.path.join(opts.weight_dir, im_name + '.npy'), allow_pickle=True)
                print(f'weight.dtype: {weights_delta.dtype}')
                # print(f'weight: {weights_delta}')
                weights_delta = torch.Tensor(weights_delta).to(device)
                print(f'weight.shape: {weights_delta.shape}')


                tic = time.time()
                result_batch = run_inversion_test(input_latent, opts, weights_delta)
                # result_batch, result_latents, result_deltas, _ = run_inversion(input_cuda, net, opts,
                #                                                             return_intermediate_results=False)
                toc = time.time()
                global_time.append(toc - tic)



                input_im = tensor2im(result_batch)

                result = np.array(input_im.resize(resize_amount))

                # save individual outputs
                save_dir = os.path.join(out_path_results)
                os.makedirs(save_dir, exist_ok=True)
                result.save(os.path.join(save_dir, im_name + 'jpg'))

                # save coupled image with side-by-side results
                # Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)



if __name__ == '__main__':
    run()
