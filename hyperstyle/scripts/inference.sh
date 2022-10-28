#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
--latents \
--latent_dir='/mnt/pami23/yfyuan/EXP/LATENT/hyperstyle/celeba_hq_test_save_weight' \
--load_w_encoder \
--save_weight_deltas \
--exp_dir=/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024_save_weight




CUDA_VISIBLE_DEVICES=6 python scripts/inference_test.py \
--load_w_encoder \
--exp_dir=/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_my \
--latent_dir='/mnt/pami23/yfyuan/EXP/LATENT/hyperstyle/celeba_hq_test' \
--weight_dir='/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024_save_weight/weight_deltas'



