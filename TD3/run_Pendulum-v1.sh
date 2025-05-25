#!/bin/bash
#SBATCH --job-name=pendulum_training
#SBATCH --output=TD3_Pendulum-v1_%j.out
#SBATCH --error=TD3_Pendulum-v1_%j.err
#SBATCH --time=04:00:00            # Adjust time as needed
#SBATCH --mem=4G                   # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1


SEED=${1:-0}
EXPL_NOISE=${2:-0.1}
POLICY_NOISE=${3:-0.2}
NOISE_CLIP=${4:-0.5}
TAU=${5:-0.005}
BATCH_SIZE=${6:-256}

# Activate your environment
source /home/maillet/venvs/env_td3/bin/activate

# Move to your project directory
cd /home/maillet/RL-project/TD3

# Run your code
python main.py --policy "TD3" --env Pendulum-v1 --max_timesteps 50000 --seed "$SEED" --expl_noise "$EXPL_NOISE" --policy_noise "$POLICY_NOISE" --noise_clip "$NOISE_CLIP" --tau "$TAU" --batch_size "$BATCH_SIZE"
