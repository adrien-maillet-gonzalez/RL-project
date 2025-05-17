#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --output=PPO_MountainCarContinuous-v0_%j.out
#SBATCH --error=PPO_MountainCarContinuous-v0_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                                # (If you need a GPU - remove this if CPU only)

# Read seed from command line, default to 0 if not provided
SEED=${1:-0}

cd /home/maillet/RL-project/PPO/baselines-master

python -m baselines.run --alg=ppo2 --env=MountainCarContinuous-v0 --num_timesteps=1e7   --ent_coef=0.02 --lr=5e-2 --gamma=0.98 --nsteps=4096   --nminibatches=8 --vf_coef=0.7 --seed=${SEED}
