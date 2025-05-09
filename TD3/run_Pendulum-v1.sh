#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --output=TD3_Pendulum-v1_seed-37_%j.out
#SBATCH --error=TD3_Pendulum-v1_seed-37_%j.err
#SBATCH --time=04:00:00            # Adjust time as needed
#SBATCH --mem=4G                   # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1

# Activate your environment
source /home/maillet/venvs/env_rl/bin/activate

# Move to your project directory
cd /home/maillet/TD3-original-paper/TD3

# Run your code
python main.py --policy "TD3" --env Pendulum-v1 --seed 37
