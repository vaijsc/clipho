#!/bin/bash

# Define the maximum number of retries
MAX_RETRIES=25
RETRY_COUNT=0

# Command to run the training script
TRAINING_CMD="/lustre/scratch/client/movian/applied/users/dungnn28/miniconda3/envs/up2date_cu124/bin/torchrun \
    --nproc_per_node=8 --nnode=1 \
    src/run_vision_encoder.py \
    --output_dir saves/google_siglip-base-patch16-256-multilingual_phobert_syllable_base_512_images_27m_CC3M_stage_2_new \
    --model_name_or_path saves/google_siglip-base-patch16-256-multilingual_phobert_syllable_base_512_images_27m_CC3M_stage_1/checkpoint-16109 \
    --freeze_vision_model=True \
    --freeze_text_model=False \
    --dataset_name image_27M_CC3M \
    --image_data_paths /lustre/scratch/client/movian/asset/datasets/thinhphp1/text2img/vidata/images_27M,pixparse/cc3m-wds \
    --image_column image \
    --caption_data_paths output/vie_captions/captions_images_27M_refined_prompts.jsonl,output/vie_captions/captions_CC3M.jsonl \
    --caption_column caption \
    --remove_unused_columns=False \
    --removed_indices_file data/tmp/indices/all_removed_indices_images_27M_CC3M.json \
    --custom_order=False \
    --shuffle_data=True \
    --do_train \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1024 \
    --gradient_accumulation_steps 4 \
    --enable_gradient_checkpointing=True \
    --optim adamw_torch \
    --learning_rate 5e-5 \
    --weight_decay 5e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 16 \
    --dataloader_persistent_workers=True \
    --dataloader_prefetch_factor 4 \
    --logging_first_step=True \
    --logging_steps 23 \
    --save_steps 90 \
    --use_liger_kernel=True \
    --report_to tensorboard \
    --ddp_timeout 3600 \
    --bf16=True \
    --deepspeed configs/ds_z0_config.json \
    --seed 2024"

# Function to run the training and handle errors
run_training() {
    echo "### Starting training attempt $((RETRY_COUNT + 1)) ###"

    # Run the training command
    eval $TRAINING_CMD

    # Capture the exit code of the training process
    EXIT_CODE=$?

    # If the exit code is 0 (success), exit the loop
    if [ $EXIT_CODE -eq 0 ]; then
        echo "### Training completed successfully. ###"
        exit 0
    fi

    # If SIGKILL (exit code 137 or -9), increment retry count and rerun
    if [ $EXIT_CODE -eq 1 ] || [ $EXIT_CODE -eq 9 ] || [ $EXIT_CODE -eq 137 ]; then
        echo "### Training process terminated with SIGKILL. Retrying ###"
        RETRY_COUNT=$((RETRY_COUNT + 1))
    else
        echo "### Training failed with exit code $EXIT_CODE. Stopping ###"
        exit $EXIT_CODE
    fi
}

# Main loop to retry training
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    run_training
done

echo "### Training failed after $MAX_RETRIES attempts. ###"
exit 1
