#!/bin/bash

# Define the maximum number of retries
MAX_RETRIES=20
RETRY_COUNT=0

# Command to run the training script
TRAINING_CMD="torchrun --nproc_per_node=8 --nnode=1 src/run_vision_encoder.py \
    --output_dir saves/openai_clip-vit-base-patch16_phobert_syllable_stage-1_no_gemini \
    --model_name_or_path models/openai_clip-vit-base-patch16_phobert_syllable \
    --freeze_vision_model=True \
    --dataset_name image_27M \
    --dataset_config_name all \
    --image_data_dir /lustre/scratch/client/vinai/users/thinhphp1/text2img/vidata/images_27M/ \
    --image_column image \
    --caption_data_dir output/vie_captions/ \
    --caption_column caption \
    --remove_unused_columns=False \
    --removed_indices_file data/tmp/indices/all_removed_indices_new.json \
    --custom_order=False \
    --shuffle_data=True \
    --do_train \
    --num_train_epochs 20 \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --enable_gradient_checkpointing=True \
    --optim adamw_torch \
    --learning_rate 5e-5 \
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
    --logging_steps 129 \
    --save_steps 643 \
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
