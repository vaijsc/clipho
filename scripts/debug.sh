#!/bin/bash -e
#SBATCH --job-name=debug_2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64GB
#SBATCH --partition=applied
#SBATCH --output=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/debug/debug_2.txt
#SBATCH --error=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/debug/debug_2.txt

module purge
module load python/miniconda3/miniconda3/
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/miniconda3/envs/align_llm
cd /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM

python src/debug.py \
    --model_path tmp/models/google_siglip-base-patch16-384_phobert_syllable_base_512_image_27M_all_stage-1_new/checkpoint-7136 \
    --start_step 8771 \
    --end_step 8950 \
    --start_index 409600 \
    --num_samples 204800 \
    --batch_size 512
