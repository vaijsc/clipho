#!/bin/bash -e
#SBATCH --job-name=vlm_cap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=60GB
#SBATCH --partition=applied
#SBATCH --output=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/gen_captions_images_27M_MiniCPM-V-2_6_system-prompt_detail_0_1m.txt
#SBATCH --error=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/gen_captions_images_27M_MiniCPM-V-2_6_system-prompt_detail_0_1m.txt

module purge
module load python/miniconda3/miniconda3/
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/miniconda3/envs/align_llm
cd /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM

bash -c'python src/gen_caption_vllm.py \
--model_id openbmb/MiniCPM-V-2_6 \
--data_name images_27M_MiniCPM-V-2_6_system-prompt_detail_0_1m \
--start_index 0 \
--end_index 1000000 \
--use_system_prompt \
--caption_type 0 \
--batch_size 512 \
--max_model_len 2048
'