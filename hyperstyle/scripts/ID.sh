#!/bin/bash
Hyperstyle:
---------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/calc_id_loss_parallel.py \
--output_path='/mnt/pami23/yfyuan/EXP/TEST_hyperstyle_rec_1024/inference_results/4' \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'

CUDA_VISIBLE_DEVICES=0 python scripts/calc_id_loss_parallel.py \
--output_path='/mnt/pami23/yfyuan/EXP/TEST_lattrans/alone_edited/hyperstyle/all_celeba_test_Smiling/Smiling_1.5' \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'


e4e:
---------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/calc_id_loss_parallel.py \
--output_path='/mnt/pami23/yfyuan/EXP/LATENT/e4e/celeba_hq_test/inference_results' \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'

CUDA_VISIBLE_DEVICES=0 python scripts/calc_id_loss_parallel.py \
--output_path='/mnt/pami23/yfyuan/EXP/TEST_lattrans/alone_edited/e4e/all_celeba_test_Smiling/Smiling_1.5' \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'


pSp:
---------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/calc_id_loss_parallel.py \
--output_path='/mnt/pami23/yfyuan/EXP/LATENT/pSp/celeba_hq_test/inference_results' \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'

CUDA_VISIBLE_DEVICES=0 python scripts/calc_id_loss_parallel.py \
--output_path='/mnt/pami23/yfyuan/EXP/TEST_lattrans/alone_edited/pSp/all_celeba_test_Smiling/Smiling_1.0' \
--gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'



