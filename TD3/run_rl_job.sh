#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --output=MountainCarContinuous-v0_%j.out
#SBATCH --error=MountainCarContinuous-v0_%j.err
#SBATCH --time=01:00:00            # Adjust time as needed
#SBATCH --mem=8G                   # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1               # Only if GPU is required

# Activate your environment
source /home/maillet/venvs/env_rl/bin/activate

# Move to your project directory
cd /home/maillet/TD3-original-paper/TD3

# Run your code
python main.py --env MountainCarContinuous-v0
