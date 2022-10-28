#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python scripts/calc_losses_on_images.py \
--metrics lpips,l2,msssim \
--output_path=/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024/inference_results \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'


