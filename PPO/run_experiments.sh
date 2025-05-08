#!/bin/bash

# This script runs all three PPO variants on both environments
# For each algorithm and environment, it runs with 5 different seeds

# MountainCarContinuous-v0 Environment
# PPO-Base
python main.py --policy PPO-Base --env MountainCarContinuous-v0 --seed 0 --save_model
python main.py --policy PPO-Base --env MountainCarContinuous-v0 --seed 1 --save_model
python main.py --policy PPO-Base --env MountainCarContinuous-v0 --seed 2 --save_model
python main.py --policy PPO-Base --env MountainCarContinuous-v0 --seed 3 --save_model
python main.py --policy PPO-Base --env MountainCarContinuous-v0 --seed 4 --save_model

# PPO-Clip
python main.py --policy PPO-Clip --env MountainCarContinuous-v0 --seed 0 --save_model
python main.py --policy PPO-Clip --env MountainCarContinuous-v0 --seed 1 --save_model
python main.py --policy PPO-Clip --env MountainCarContinuous-v0 --seed 2 --save_model
python main.py --policy PPO-Clip --env MountainCarContinuous-v0 --seed 3 --save_model
python main.py --policy PPO-Clip --env MountainCarContinuous-v0 --seed 4 --save_model

# PPO-KL
python main.py --policy PPO-KL --env MountainCarContinuous-v0 --seed 0 --save_model
python main.py --policy PPO-KL --env MountainCarContinuous-v0 --seed 1 --save_model
python main.py --policy PPO-KL --env MountainCarContinuous-v0 --seed 2 --save_model
python main.py --policy PPO-KL --env MountainCarContinuous-v0 --seed 3 --save_model
python main.py --policy PPO-KL --env MountainCarContinuous-v0 --seed 4 --save_model

# Pendulum-v1 Environment
# PPO-Base
python main.py --policy PPO-Base --env Pendulum-v1 --seed 0 --save_model
python main.py --policy PPO-Base --env Pendulum-v1 --seed 1 --save_model
python main.py --policy PPO-Base --env Pendulum-v1 --seed 2 --save_model
python main.py --policy PPO-Base --env Pendulum-v1 --seed 3 --save_model
python main.py --policy PPO-Base --env Pendulum-v1 --seed 4 --save_model

# PPO-Clip
python main.py --policy PPO-Clip --env Pendulum-v1 --seed 0 --save_model
python main.py --policy PPO-Clip --env Pendulum-v1 --seed 1 --save_model
python main.py --policy PPO-Clip --env Pendulum-v1 --seed 2 --save_model
python main.py --policy PPO-Clip --env Pendulum-v1 --seed 3 --save_model
python main.py --policy PPO-Clip --env Pendulum-v1 --seed 4 --save_model

# PPO-KL
python main.py --policy PPO-KL --env Pendulum-v1 --seed 0 --save_model
python main.py --policy PPO-KL --env Pendulum-v1 --seed 1 --save_model
python main.py --policy PPO-KL --env Pendulum-v1 --seed 2 --save_model
python main.py --policy PPO-KL --env Pendulum-v1 --seed 3 --save_model
python main.py --policy PPO-KL --env Pendulum-v1 --seed 4 --save_model