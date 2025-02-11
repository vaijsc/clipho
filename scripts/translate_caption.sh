#!/bin/bash -e
#SBATCH --job-name=tran1617
#SBATCH --output=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/translate_captions_images_27M_MiniCPM-V-2_6_system-prompt_detail_16m_17m.txt
#SBATCH --error=/lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/VLM/logs/translate_captions_images_27M_MiniCPM-V-2_6_system-prompt_detail_16m_17m.txt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64GB
#SBATCH --nodelist=sdc2-hpc-dgx-a100-010
#SBATCH --partition=applied


module purge
module load python/miniconda3/miniconda3/
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/miniconda3/envs/align_llm
cd /lustre/scratch/client/scratch/phogpt/phogptgroup/dungnn28/trans_tools

bash trans.sh 0 captions_images_27M_MiniCPM-V-2_6_system-prompt_detail_16m_17m 32