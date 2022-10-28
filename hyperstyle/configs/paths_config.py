ckpt_paths = {
    'pSp':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/psp_ffhq_encode.pt',
    'e4e':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/e4e_ffhq_encoder.pt',
    'hyperstyle':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/hyperstyle_ffhq.pt'
}

dataset_paths = {
    'celeba_train':
    '/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/train/images',
    #'/mnt/pami23/yfyuan/DATASET/CelebAMask-HQ/CelebA-HQ-img',
    'celeba_test':
    '/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images',
    # 'celeba_test': '/mnt/pami23/yfyuan/DATASET/CelebA_hq_attr_split/Gender/test/Female',
    # 'celeba_test': '/mnt/pami23/yfyuan/DATASET/CelebA_hq_attr_split/Smiling/test/unsmiling',
    'celeba_test_w_inv': '',
    'celeba_test_w_latents': '',
    'ffhq': '/mnt/pami23/yfyuan/DATASET/FFHQ_1024x1024',
    'ffhq_w_inv': '',
    'ffhq_w_latents': '',
    'afhq_wild_train': '',
    'afhq_wild_test': '',
    'cars_train': '',
    'cars_test': '',
}

model_paths = {
    # models for backbones and losses
    'ir_se50':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/model_ir_se50.pth',
    'resnet34':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/resnet34-333f7ec4.pth',
    'moco':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/moco_v2_800ep_pretrain.pt',
    # stylegan2 generators
    'stylegan_ffhq':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/stylegan2-ffhq-config-f.pt',
    'stylegan_cars':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/stylegan2-car-config-f.pt',
    'stylegan_ada_wild':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/afhqwild.pt',  #TODO
    # model for face alignment
    'shape_predictor':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/shape_predictor_68_face_landmarks.dat',
    # models for ID similarity computation
    'curricular_face':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet':
    '/mnt/pami23/yfyuan/EXP/ICASSP2022/pretrain_model/e4e_pretrained_models/mtcnn/onet.npy',
    # WEncoders for training on various domains
    'faces_w_encoder':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/faces_w_encoder.pt',
    'cars_w_encoder':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/cars_w_encoder.pt',  #TODO
    'afhq_wild_w_encoder':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/afhq_wild_w_encoder.pt',  #TODO
    # models for domain adaptation
    'restyle_e4e_ffhq':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/restyle_e4e_ffhq_encode.pt',
    'stylegan_pixar':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/pixar.pt',
    'stylegan_toonify':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/ffhq_cartoon_blended.pt',
    'stylegan_sketch':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/sketch.pt',  #TODO
    'stylegan_disney':
    '/mnt/pami23/yfyuan/PRETRAIN_MODEL/HyperStyle/disney_princess.pt'  #TODO
}

edit_paths = {
    'age':
    '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/interfacegan_directions/age.pt',
    'smile':
    '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/interfacegan_directions/smile.pt',
    'pose':
    '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/interfacegan_directions/pose.pt',
    'male':
    '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/interfacegan_directions/male.pt',
    'eyeglasses':
    '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/interfacegan_directions/eyeglasses.pt',
    'cars': 'editing/ganspace_directions/cars_pca.pt',
    'styleclip': {
        'delta_i_c':
        '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/styleclip/global_directions/ffhq/fs3.npy',
        's_statistics':
        '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/styleclip/global_directions/ffhq/S_mean_std',
        'templates':
        '/home/stma/workspace/lattrans_hyperstyle/hyperstyle/editing/styleclip/global_directions/templates.txt'
    }
}