#!/bin/bash

# Define the maximum number of retries
MAX_RETRIES=20
RETRY_COUNT=0

# Command to run the training script
TRAINING_CMD="/lustre/scratch/client/movian/applied/users/dungnn28/miniconda3/envs/up2date_cu124/bin/torchrun \
    --nproc_per_node=8 --nnode=1 \
    src/run_vision_encoder.py \
    --output_dir models/google_siglip2-base-patch16-256_no_dual_projection_stage_1 \
    --model_name_or_path google/siglip2-base-patch16-256 \
    --use_vision_text_dual_encoder=False \
    --use_flash_attn_2=True \
    --freeze_vision_model=True \
    --unfreeze_vision_modules encoder.layers.11 encoder.layers.10 post_layernorm head \
    --freeze_text_model=True \
    --unfreeze_text_modules encoder.layers.11 encoder.layers.10 final_layer_norm head \
    --dataset_name image_27M_CC3M \
    --image_data_paths /lustre/scratch/client/movian/asset/datasets/thinhphp1/text2img/vidata/images_27M,pixparse/cc3m-wds \
    --image_column image \
    --caption_data_paths output/vie_captions/captions_images_27M_refined_prompts.jsonl,output/vie_captions/captions_CC3M.jsonl \
    --caption_column caption \
    --remove_unused_columns=False \
    --removed_indices_file data/tmp/indices/all_removed_indices_images_27M_CC3M.json \
    --custom_order=False \
    --shuffle_data=True \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --do_train \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1024 \
    --gradient_accumulation_steps 4 \
    --enable_gradient_checkpointing=True \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers=True \
    --dataloader_prefetch_factor 2 \
    --logging_first_step=True \
    --logging_steps 23 \
    --save_steps 89 \
    --use_liger_kernel=True \
    --report_to tensorboard \
    --ddp_timeout 3600 \
    --bf16=True \
    --deepspeed configs/ds_z2_config.json \
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
