#!/bin/bash

#SBATCH --job-name=3DG-dnerf              # Job name
#SBATCH --time=12:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1             # must use this GPU, since pytorch3d relied on it
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate 3dg

python train.py --data-dir ./data/dnerf --exp-name dnerf --dataset bouncingballs --dataset-type dnerf
