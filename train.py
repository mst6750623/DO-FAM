import torch
import os
import yaml
import glob
import argparse
import numpy as np
import torch.utils.data as data
from trainer import Trainer
from dataset import LatentDataset
from tqdm import tqdm
from configs.path_config import ckpt_paths, data_paths


def main(opts):
    # Celeba attribute list
    '''attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
                  'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
                  'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
                  'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
                  'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
                  'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
                  'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
                  'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}'''
    attr_dict = {
        #'Arched_Eyebrows': 1,
        #'Bald': 4,
        #'Bangs': 5,
        #'Black_Hair': 8,
        #'Blond_Hair': 9,
        #'Double_Chin': 14,
        'Eyeglasses': 15,
        #'Heavy_Makeup': 18,
        'Male': 20,
        #'Mouth_Slightly_Open': 21,
        #'Mustache': 22,
        #'Narrow_Eyes': 23,
        #'No_Beard': 24,
        #'Pale_Skin': 26,
        'Smiling': 31,
        #'Wearing_Lipstick': 36,
        'Young': 39
    }
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    log_dir = os.path.join(opts.log_path, 'checkpoint')
    os.makedirs(log_dir, exist_ok=True)

    config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'),
                       Loader=yaml.FullLoader)
    attr_list = config['attr'].split(',')
    batch_size = config['batch_size']
    epochs = config['epochs']

    test_latents_list = [
        glob.glob1(opts.test_latent_path, ext) for ext in ['*pt']
    ]
    test_latents_list = [
        item for sublist in test_latents_list for item in sublist
    ]
    test_latents_list.sort()
    test_w = test_latents_list[:10]

    weights_list = [
        glob.glob1(opts.test_weights_delta_path, ext) for ext in ['*npy']
    ]
    weights_list = [item for sublist in weights_list for item in sublist]
    weights_list.sort()

    weights_list = weights_list[:10]
    dataset_A = LatentDataset(opts.latent_path,
                              opts.label_file,
                              training_set=True)
    loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    print(len(dataset_A))
    print('Start training!')
    print('config:', opts.config)
    print('cls_path:', opts.classifier_model_path)
    print('latent_path:', opts.latent_path)
    print('log_path:', opts.log_path)

    test_w_temp = torch.tensor(
        torch.load(os.path.join(opts.test_latent_path, test_w[0]))).to(device)
    weights_temp = np.load(os.path.join(opts.test_weights_delta_path,
                                        weights_list[0]),
                           allow_pickle=True)
    for attr in attr_dict.items():
        attr_name = attr[0]
        attr_num = attr[1]
        total_iter = 0
        #attr_num = attr_dict[attr]
        print(attr_name, attr_num)
        # Initialize trainer
        trainer = Trainer(config, opts, attr_num, attr_name)
        trainer.initialize(opts.stylegan_model_path,
                           opts.classifier_model_path)
        trainer.to(device)

        for n_epoch in range(epochs):
            print(f'epoch: {n_epoch}')
            for n_iter, list_A in enumerate(tqdm(loader_A)):

                w_A, label_A = list_A
                w_A, label_A = w_A.to(device), label_A.to(device)
                trainer.update(w_A, None, n_iter)

                if (total_iter + 1) % config['log_iter'] == 0:
                    trainer.log_loss(total_iter)

                if (total_iter) % config['image_log_iter'] == 0:

                    trainer.log_image(test_w_temp.unsqueeze(0), total_iter,
                                      weights_temp)
                total_iter += 1

            trainer.scheduler.step()
        trainer.save_model(log_dir)

    print('Oops! Training finished!')


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='DOLL',
                        help='Path to the config file.')
    parser.add_argument('--latent_path',
                        type=str,
                        default=data_paths['train_latent'],
                        help='dataset path')
    parser.add_argument('--test_latent_path',
                        type=str,
                        default=data_paths['test_latent'],
                        help='test dataset path')
    parser.add_argument('--test_weights_delta_path',
                        type=str,
                        default=data_paths['test_weights_delta'],
                        help='test weights delta path')
    parser.add_argument('--label_file',
                        type=str,
                        default=data_paths['label_file'],
                        help='label file path')
    parser.add_argument('--stylegan_model_path',
                        type=str,
                        default=ckpt_paths['hyperstyle'],
                        help='stylegan model path')
    parser.add_argument('--classifier_model_path',
                        type=str,
                        default=ckpt_paths['classifier'],
                        help='pretrained attribute classifier')
    parser.add_argument('--log_path',
                        type=str,
                        default='logs/',
                        help='log file path')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='',
                        help='checkpoint file path')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='use multiple gpus')
    parser.add_argument('--extra_init',
                        action='store_true',
                        help='decide whether use param init or not')
    opts = parser.parse_args()
    main(opts)
