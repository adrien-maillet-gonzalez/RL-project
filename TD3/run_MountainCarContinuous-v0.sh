#!/bin/bash
#SBATCH --job-name=MountainCarContinuous_training
#SBATCH --output=TD3_MountainCarContinuous-v0_expl_noise_0.7_%j.out
#SBATCH --error=TD3_MountainCarContinuous-v0_expl_noise_0.7_%j.err
#SBATCH --time=02:00:00            # Adjust time as needed
#SBATCH --mem=4G                  # Adjust memory as needed
#SBATCH --cpus-per-task=4          # Adjust cores as needed
#SBATCH --gres=gpu:1               # (If you need a GPU - remove this if CPU only)


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
python main.py --policy "TD3" --env MountainCarContinuous-v0 --max_timesteps 500000 --seed "$SEED" --expl_noise "$EXPL_NOISE" --policy_noise "$POLICY_NOISE" --noise_clip "$NOISE_CLIP" --tau "$TAU" --batch_size "$BATCH_SIZE"
