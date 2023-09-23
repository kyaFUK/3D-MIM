#!/bin/bash
python -u run.py \
    --is_training=True \
    --dataset_name sn \
    --train_data_paths ./data/sn/SampleData.npz \
    --valid_data_paths ./data/sn/SampleData.npz \
    --save_dir checkpoints/Sample \
    --gen_frm_dir results/Sample \
    --model_name mim \
    --allow_gpu_growth=True \
    --img_channel 1 \
    --img_width 32 \
    --input_length 1 \
    --total_length 20 \
    --max_iterations 1\
    --display_interval 1 \
    --test_interval 12 \
    --snapshot_interval 1 \
    --num_hidden 32,32,32,32 \
    --batch_size 1 \
    --patch_size 1 \
    --num_save_samples 12 \
    |& tee sample.log