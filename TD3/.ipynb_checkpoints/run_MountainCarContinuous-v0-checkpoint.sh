#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --output=TD3_MountainCarContinuous-v0_seed-2_expl_noise_0.7_%j.out
#SBATCH --error=TD3_MountainCarContinuous-v0_seed-2_expl_noise_0.7_%j.err
#SBATCH --time=02:00:00            # Adjust time as needed
#SBATCH --mem=4G                  # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1               # (If you need a GPU - remove this if CPU only)

# Activate your environment
source /home/maillet/venvs/env_td3/bin/activate

# Move to your project directory
cd /home/maillet/RL-project/TD3

# Run your code
python main.py --policy "TD3" --env MountainCarContinuous-v0 --seed 2 --expl_noise 0.7
