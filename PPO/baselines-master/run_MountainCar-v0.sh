#!/bin/bash
#SBATCH --job-name=rl_training_PPO_MountainCar-v0
#SBATCH --output=PPO_MountainCar-v0_%j.out
#SBATCH --error=PPO_MountainCar_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Read seed from command line, default to 0 if not provided
SEED=${1:-0}

cd /home/maillet/RL-project/PPO/baselines-master
# cd C:/Users/franc/Documents/Cours/EPFL/M2/Reinforcement Learning/Final_project/RL-project/PPO/baselines-master

# python -m baselines.run --alg=ppo2 --env=MountainCar-v0 --num_timesteps=1e6 --lr=2e-4 --nminibatches=64 --noptepochs=4  --seed=49 --save_path=models/MountainCar_ppo2_seed49
# python -m baselines.run --alg=ppo2 --env=MountainCarContinuous-v0 --num_timesteps=10e6   --ent_coef=0.02 --lr=5e-2 --gamma=0.98 --nsteps=4096   --nminibatches=8 --vf_coef=0.7 --seed=42
python -m baselines.run --alg=ppo2 --env=MountainCar-v0 --num_timesteps=1.5e6 --lr=2e-3 --nminibatches=64 --noptepochs=4  --seed=54 --save_path=models/MountainCar_ppo2_seed54

# Load the trained model
python -m baselines.run --alg=ppo2 --env=MountainCar-v0 --num_timesteps=0 --load_path=models/MountainCarDiscrete_ppo2_seed_$SEED --play

# runed using seeds 48,49,50
