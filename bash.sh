
python main.py \
       --phase train \
       --dataset_name 'Cityscapes' \
       --dataset 'data/Cityscapes' \
       --result_dir 'sample' \
       --batch_size 2 \
       --epoch 10 \
       --target_class 13 \
       --class_num 19 \
       --is_layout_real True \
       --save_freq 1  \
       --theta_rec_weight 100 \
       --affine_weight 1 \
       --layout_weight 1 \
       --device 'cpu'


python main.py \
       --phase test \
       --dataset_name 'Cityscapes' \
       --dataset 'data/Cityscapes' \
       --batch_size 1 \
       --result_dir 'sample' \
       --target_class 13 \
       --class_num 19 \
       --is_layout_real True \
       --device 'cpu'



python main.py \
       --phase train \
       --dataset_name 'Cityscapes' \
       --dataset 'data/Cityscapes' \
       --result_dir 'sample' \
       --batch_size 2 \
       --epoch 10 \
       --target_class 13 \
       --class_num 19 \
       --is_layout_real True \
       --save_freq 10  \
       --theta_rec_weight 100 \
       --affine_weight 1 \
       --layout_weight 1 \
       --device 'cpu'


python main.py \
       --phase test \
       --dataset_name 'Cityscapes' \
       --dataset 'data/Cityscapes' \
       --batch_size 1 \
       --result_dir 'sample' \
       --target_class 13 \
       --class_num 19 \
       --is_layout_real True \
       --device 'cpu'