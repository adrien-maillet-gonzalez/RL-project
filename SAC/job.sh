#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --output=rl_training_%j.out
#SBATCH --error=rl_training_%j.err
#SBATCH --time=07:00:00            # Adjust time as needed
#SBATCH --mem=32G                   # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1               # (If you need a GPU - remove this if CPU only)

# Load modules if needed (you may not need this if your env handles it)
# module load python/3.X.X



module load gcc/11.3.0
# Load CUDA 11.8 and its corresponding cuDNN 8.7
module load cuda/11.8.0
module load cudnn/8.7.0.84-11.8
#module list
# Activate your environment


#mamba activate sac-tfagents


# Move to your project directory
cd /home/pinoprie/new_rl/RL-project/SAC

# Run your code
python sac.py
