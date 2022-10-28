#!/bin/bash
CUDA_VISIBLE_DEVICES=0 pytorch-fid /mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images \
/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024/inference_results/4

CUDA_VISIBLE_DEVICES=0 pytorch-fid /mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images \
/mnt/pami23/yfyuan/EXP/LATENT/e4e/inference_results
CUDA_VISIBLE_DEVICES=6 pytorch-fid /mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images /mnt/pami23/yfyuan/EXP/TEST_lattrans/alone_edited/hyperstyle/unsmiling2smiling/Smiling_1.0