#!/bin/bash
#SBATCH --job-name=rl_training_PPO_CartPole-v1
#SBATCH --output=PPO_CartPole-v1_%j.out
#SBATCH --error=PPO_CartPole-v1_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Read seed from command line, default to 0 if not provided
SEED=${1:-0}

# Move to your project directory
cd /home/maillet/RL-project/PPO/baselines-master

# Run your code with the given seed
python -m baselines.run --alg=ppo2 --env=CartPole-v1 --network=mlp --num_timesteps=1e7 --seed=$SEED
