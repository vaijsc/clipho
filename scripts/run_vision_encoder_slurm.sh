#!/bin/bash -e
#SBATCH --job-name=clip_b16
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=100GB
#SBATCH --partition=applied
#SBATCH --output=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/run_vision_encoder/openai_clip-vit-base-patch16_phobert_syllable_base_512_image_27M_all.txt
#SBATCH --error=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/openai_clip-vit-base-patch16_phobert_syllable_base_512_image_27M_all.txt

module purge
module load python/miniconda3/miniconda3/
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/miniconda3/envs/align_llm
cd /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

torchrun --nproc_per_node $GPUS_PER_NODE \
    --nnode $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/run_vision_encoder.py \
    --output_dir saves/openai_clip-vit-base-patch16_phobert_syllable_base_512_image_27M_all \
    --model_name_or_path models/openai_clip-vit-base-patch16_phobert_syllable_base_512 \
    --freeze_vision_model=True \
    --dataset_name image_27M \
    --dataset_config_name all \
    --image_data_dir /lustre/scratch/client/vinai/users/thinhphp1/text2img/vidata/images_27M/ \
    --image_column image \
    --caption_data_dir output/vie_captions/ \
    --caption_column caption \
    --remove_unused_columns=False \
    --selected_indices_file data/tmp/indices/gemini-1.5-flash_caption_indices.json \
    --removed_indices_file data/tmp/indices/invalid_caption_indices.json \
    --custom_order=True \
    --do_train \
    --num_train_epochs 30 \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --enable_gradient_checkpointing=True \
    --optim adamw_hf \
    --learning_rate 1.25e-4 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 16 \
    --dataloader_persistent_workers=True \
    --dataloader_prefetch_factor 4 \
    --logging_first_step=True \
    --logging_steps 268 \
    --save_steps 1339 \
    --use_liger_kernel=True \
    --report_to tensorboard \
    --ddp_timeout 3600 \
    --bf16=True \
    --deepspeed configs/ds_z2_config.json
