#!/bin/bash -l

d_model_arr=(128 64 64 256 128)
model_dim_arr=(128 64 64 256 128)
dim_feedforward_arr=(512 256 128 512 512)
en_layers_arr=(1 2 2 1 2)
dec_layers_arr=(1 1 1 1 1)
dec_layers_recon_arr=(1 1 1 1 1)
gamma_arr=(0.8 0.8 0.8 0.5 0.8)
sparsity_arr=(0.3 0.1 0.1 0.3 0.2)
epsilon_arr=(0.1 0.3 0.05 0.3 0.3)
w_pred_kl_arr=(1 1 1 1 1)
w_social_arr=(1 1 1 1 1)
epochs_thresh_arr=(10 20 20 20 20)
step_size_arr=(10 20 20 10 40)
D_arr=(0.5 0.5 0.5 0.5 0.5)
dataset_arr=(eth hotel univ zara1 zara2)

python train.py --dataset ${dataset_arr[$SLURM_ARRAY_TASK_ID]} \
    --w_pred_kl ${w_pred_kl_arr[$SLURM_ARRAY_TASK_ID]} \
    --w_social ${w_social_arr[$SLURM_ARRAY_TASK_ID]} \
    --compute_sample \
    --min_mean 0 \
    --c 1e-4 \
    --min_clip_cvae 0 \
    --min_clip_vae 0 \
    --d_model ${d_model_arr[$SLURM_ARRAY_TASK_ID]} \
    --model_dim ${model_dim_arr[$SLURM_ARRAY_TASK_ID]} \
    --num_params 32\
    --en_layers ${en_layers_arr[$SLURM_ARRAY_TASK_ID]} \
    --dec_layers ${dec_layers_arr[$SLURM_ARRAY_TASK_ID]} \
    --dec_layers_recon ${dec_layers_recon_arr[$SLURM_ARRAY_TASK_ID]} \
    --dim_feedforward ${dim_feedforward_arr[$SLURM_ARRAY_TASK_ID]} \
    --nhead 8 \
    --epsilon ${epsilon_arr[$SLURM_ARRAY_TASK_ID]} \
    --k 20 \
    --every_few 10 \
    --epochs_thresh ${epochs_thresh_arr[$SLURM_ARRAY_TASK_ID]} \
    --sample_num 20 \
    --num_epochs 150 \
    --lr 1e-4 \
    --scheduler_type steplr \
    --choice_recon random \
    --min_lr 0 \
    --max_lr 0 \
    --initial_lr 0 \
    --warmup 0 \
    --sparsity ${sparsity_arr[$SLURM_ARRAY_TASK_ID]} \
    --D ${D_arr[$SLURM_ARRAY_TASK_ID]} \
    --step_size ${step_size_arr[$SLURM_ARRAY_TASK_ID]} \
    --gamma ${gamma_arr[$SLURM_ARRAY_TASK_ID]}


wait