#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=20
gpus=4
lr=0.000005
encoder=vitl
dataset=vkitti
img_size=322
min_depth=0.1
max_depth=20 # 80 for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/real_120
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
