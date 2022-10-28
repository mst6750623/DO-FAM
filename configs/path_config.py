ckpt_paths = {
    'hyperstyle':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/hyperstyle_ffhq.pt',
    'classifier':
    '/home/stma/workspace/yyf-latent-transformer/pretraining/checkpoint/001/latent_classifier_epoch_50.pth'
}
model_paths = {
    'Eyeglasses':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Eyeglasses.pth.tar',
    'Gender':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Male.pth.tar',
    'Smiling':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Smiling.pth.tar',
    'Age':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Young.pth.tar'
}
data_paths = {
    'train_latent': '/mnt/pami23/stma/EXP/hyperstyle/hyperstyle_latents.pt',
    'train_weights_delta': '',
    'test_latent': '/mnt/pami23/yfyuan/EXP/LATENT/hyperstyle/celeba_hq_test/',
    'test_weights_delta':
    '/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024_save_weight/weight_deltas/',
    'label_file': '/home/stma/workspace/VR-FAM/CelebAMask_anno_sorted.npy',
}
