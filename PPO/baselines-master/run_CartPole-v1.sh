#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --output=PPO_CartPole-v1_seed-3_%j.out
#SBATCH --error=PPO_CartPole-v1_seed-3_%j.err
#SBATCH --time=02:00:00                             # Adjust time as needed
#SBATCH --mem=4G                                    # Adjust memory as needed
#SBATCH --cpus-per-task=4                           # Adjust cores as needed
#SBATCH --gres=gpu:1                                # (If you need a GPU - remove this if CPU only)

# Move to your project directory
cd /home/maillet/RL-project/PPO/baselines-master

# Run your code
python -m baselines.run --alg=ppo2 --env=CartPole-v1 --network=mlp --num_timesteps=5e6 --seed=3
