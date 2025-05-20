#!/bin/bash
#SBATCH --job-name=rl_training_PPO_Pendulum-v0
#SBATCH --output=PPO_Pendulum-v0_%j.out
#SBATCH --error=PPO_Pendulum-v0_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Read seed from command line, default to 0 if not provided
SEED=${1:-0}

# Move to your project directory
cd /home/maillet/RL-project/PPO/baselines-master

# Run your code with the given seed
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --network=mlp --num_timesteps=1e7 --seed=$SEED

# Run the code with the given seed and save the model
python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --lr=7e-4 --num_timesteps=1.2e6 --seed=$SEED --save_path=models/pendulum_ppo2_seed${SEED} 

# Play the trained model
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --num_timesteps=0 --load_path=models/pendulum_ppo2_seed$SEED --play
