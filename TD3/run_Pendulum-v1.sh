#!/bin/bash
#SBATCH --job-name=pendulum_training
#SBATCH --output=TD3_Pendulum-v1_%j.out
#SBATCH --error=TD3_Pendulum-v1_%j.err
#SBATCH --time=04:00:00            # Adjust time as needed
#SBATCH --mem=4G                   # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1


SEED=${1:-0}


# Activate your environment
source /home/maillet/venvs/env_td3/bin/activate

# Move to your project directory
cd /home/maillet/RL-project/TD3

# Run your code
python main.py --policy "TD3" --env Pendulum-v1 --max_timesteps 100000 --seed "$SEED"
