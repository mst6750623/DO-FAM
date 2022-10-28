import os
import time
import numpy as np
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader

import sys

sys.path.append(".")
sys.path.append("..")
from tqdm import tqdm
from hyperstyle.configs import data_configs
from hyperstyle.datasets.inference_dataset import InferenceDataset
from hyperstyle.utils.inference_utils import run_inversion
from lattrans_hyperstyle.hyperstyle.utils.model_utils import load_model
from configs.path_config import ckpt_paths, data_paths


def run(test_opts):
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    os.makedirs(test_opts.latent_dir, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(
        root=opts.data_path,
        transform=transforms_dict['transform_inference'],
        opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    all_latents = []
    for input_batch in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch, result_latents, result_deltas, _ = run_inversion(
                input_cuda, net, opts, return_intermediate_results=False)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]

            if opts.latents:
                im_name = extract_filename(im_path)
                latent_dir = os.path.join(test_opts.exp_dir, "latent_codes")
                os.makedirs(latent_dir, exist_ok=True)
                latent_path = os.path.join(
                    latent_dir, 'latent_code_%06d.npy' % int(im_name))
                np.save(result_latents[i], latent_path)

            if opts.save_weight_deltas:
                weight_deltas_dir = os.path.join(test_opts.exp_dir,
                                                 "weight_deltas")
                os.makedirs(weight_deltas_dir, exist_ok=True)
                np.save(
                    os.path.join(
                        weight_deltas_dir,
                        os.path.basename(im_path).split('.')[0] + ".npy"),
                    result_deltas[i][-1])

            global_i += 1


def extract_filename(f):
    _, filename = os.path.split(f)
    fname, _ = os.path.splitext(filename)
    return fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',
                        type=str,
                        default='./logs/',
                        help='Path to experiment output directory')
    parser.add_argument('--weight_dir',
                        type=str,
                        default=None,
                        help='Path to experiment output directory')
    parser.add_argument('--checkpoint_path',
                        default=ckpt_paths['hyperstyle'],
                        type=str,
                        help='Path to HyperStyle model checkpoint')
    parser.add_argument('--data_path',
                        type=str,
                        default='',
                        help='Path to directory of images to evaluate')
    parser.add_argument(
        '--resize_outputs',
        action='store_true',
        help=
        'Whether to resize outputs to 256x256 or keep at original output resolution'
    )
    parser.add_argument("--latents",
                        action="store_true",
                        help="infer the latent codes of the directory")
    parser.add_argument('--test_batch_size',
                        default=8,
                        type=int,
                        help='Batch size for testing and inference')
    parser.add_argument('--test_workers',
                        default=8,
                        type=int,
                        help='Number of test/inference dataloader workers')
    parser.add_argument(
        '--n_images',
        type=int,
        default=None,
        help='Number of images to output. If None, run on all data')
    parser.add_argument(
        '--save_weight_deltas',
        action='store_true',
        help=
        'Whether to save the weight deltas of each image. Note: file weighs about 200MB.'
    )

    # arguments for iterative inference
    parser.add_argument(
        '--n_iters_per_batch',
        default=5,
        type=int,
        help='Number of forward passes per batch during training.')

    # arguments for loading pre-trained encoder
    parser.add_argument('--load_w_encoder',
                        action='store_true',
                        help='Whether to load the w e4e encoder.')
    parser.add_argument('--w_encoder_checkpoint_path',
                        default=ckpt_paths["faces_w_encoder"],
                        type=str,
                        help='Path to pre-trained W-encoder.')
    parser.add_argument(
        '--w_encoder_type',
        default='WEncoder',
        help='Encoder type for the encoder used to get the initial inversion')

    opts = parser.parse_args()
    run()
