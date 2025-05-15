#!/bin/bash
#SBATCH --job-name=rl_training_PPO_CartPole-v1
#SBATCH --output=PPO_MountainCar-v0_%j.out
#SBATCH --error=PPO_MountainCar-v0_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Move to your project directory
cd /home/maillet/RL-project/PPO/baselines-master

# Run your code
python -m baselines.run --alg=ppo2 --env=CartPole-v1 --network=mlp --num_timesteps=5e6 --seed=0
