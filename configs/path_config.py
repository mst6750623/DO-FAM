ckpt_paths = {
    'hyperstyle': './pretrained_models/hyperstyle_ffhq.pt',
    'classifier': './pretrained_models/latent_classifier_epoch_50.pth',
    # WEncoders for training on various domains
    'faces_w_encoder': './pretrained_models/faces_w_encoder.pt',
}
model_paths = {
    'Eyeglasses':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Eyeglasses.pth.tar',
    'Gender':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Male.pth.tar',
    'Smiling':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Smiling.pth.tar',
    'Age':
    '/home/stma/workspace/lattrans_hyperstyle/logs_woID_new_5/checkpoint/l2m_Young.pth.tar',
}
data_paths = {
    'train_latent':
    '/mnt/pami23/stma/EXP/train_29000/hyperstyle/hyperstyle_latents.pt',
    'train_weights_delta': '',
    'test_images': './test_data/origin_images/',
    'test_latent': './test_data/latent_codes/',
    'test_weights_delta': './test_data/weight_deltas/',
    'label_file':
    '/mnt/pami23/stma/EXP/train_29000/celebahq_train29000_anno.npy',
}
