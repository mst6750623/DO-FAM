echo "Calculate metrics！";
echo "CUDA devices: $1";
echo "OUT path: $2";
gt_path='/mnt/pami23/yfyuan/DATASET/celeba_hq/raw_images/test/images'
echo "GT path: $gt_path"
#FID
CUDA_VISIBLE_DEVICES=$1 pytorch-fid $gt_path $2
#LPIPS,L2,SSIM
CUDA_VISIBLE_DEVICES=$1 python calc_losses_on_images.py \
--metrics lpips,l2,msssim \
--output_path=$2 \
--gt_path=$gt_path

#ID
CUDA_VISIBLE_DEVICES=$1 python calc_id_loss_parallel.py \
--output_path=$2 \
--gt_path=$gt_path