

CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	   --phase train \
       --dataset '/local-scratch2/hang/SAC-GAN/data/aachen' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/aachen' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       2>&1 | tee ./log_4_4_1.txt


# CUDA_VISIBLE_DEVICES=0 \
# python -u main.py \
#        --phase train \
#        --dataset '/local-scratch2/hang/SAC-GAN/data/aachen' \
#        --result_dir '/local-scratch2/hang/SAC-Trans/results/aachen_trans' \
#        --epoch 50 \
#        --class_num 19 \
#        --layout_flag True \
#        --save_freq 500  \
#        2>&1 | tee ./log_4_6_1.txt


CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
       --phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/aachen' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/aachen_trans' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       2>&1 | tee ./log_4_6_1.txt


CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch/hang/SAC-GAN/data/aachen' \
       --result_dir '/local-scratch/hang/SAC-Trans/results/aachen' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       2>&1 | tee ./log_5_23_1.txt


CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch/hang/SAC-GAN/data/aachen' \
       --result_dir '/local-scratch/hang/SAC-Trans/results/aachen4' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 1 \
       2>&1 | tee ./log_5_24_1.txt


CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch/hang/SAC-GAN/data/aachen' \
       --result_dir '/local-scratch/hang/SAC-Trans/results/aachen' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 1 \
       2>&1 | tee ./log_5_24_2.txt



# GPU09
# All losses
CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 1 \
       2>&1 | tee ./log_5_24_1.txt

CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag False \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 1 

# Without VAE
CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_noVAE' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 0 \
       --latent_rec_weight 0 \
       --affine_weight 1 \
       --layout_weight 1 \
       2>&1 | tee ./log_5_24_2.txt

CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_noVAE' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 0 \
       --latent_rec_weight 0 \
       --affine_weight 1 \
       --layout_weight 1

# Without GAN
CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_noGAN' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 0 \
       --layout_weight 0 \
       2>&1 | tee ./log_5_24_3.txt

CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_noGAN' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 0 \
       --layout_weight 0

# Without affine adversarial loss
CUDA_VISIBLE_DEVICES=1 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_noaffine' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 0 \
       --layout_weight 1 \
       2>&1 | tee ./log_5_24_4.txt

CUDA_VISIBLE_DEVICES=1 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_noaffine' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 0 \
       --layout_weight 1

# Without layout adversarial loss
CUDA_VISIBLE_DEVICES=1 \
python -u main.py \
	--phase train \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_nolayout' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 0 \
       2>&1 | tee ./log_5_24_5.txt

CUDA_VISIBLE_DEVICES=1 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_nolayout' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 0


# All losses + Dlayout_ref + remove Daffine_ref
CUDA_VISIBLE_DEVICES=1 \
python -u main.py \
	--phase train \
       --dataset_name 'chair' \
       --dataset '/local-scratch2/hang/SAC-GAN/data/chair' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_ref2' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag True \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 1 \
       2>&1 | tee ./log_5_24_1.txt

CUDA_VISIBLE_DEVICES=1 \
python -u main.py \
	--phase test \
       --dataset '/local-scratch2/hang/SAC-GAN/data/cityscapes' \
       --result_dir '/local-scratch2/hang/SAC-Trans/results/cityscapes_ref' \
       --epoch 10 \
       --class_num 19 \
       --layout_flag False \
       --save_freq 500  \
       --rec_weight 100 \
       --latent_rec_weight 0.05 \
       --affine_weight 1 \
       --layout_weight 1 