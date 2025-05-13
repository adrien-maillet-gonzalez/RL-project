#!/bin/bash

# Create conda environment for RL project
conda create -n RL_proj python=3.8 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate RL_proj

# Install required packages
pip install -r requirements.txt

echo "Environment RL_proj has been created and packages have been installed."
echo "To activate the environment, run: conda activate RL_proj"