#!/bin/bash

#SBATCH --job-name=colmap_dynerf                   # Job name
#SBATCH --time=1:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1             # must use this GPU, since pytorch3d relied on it
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate colmap

python preprocess_dynerf_unified.py \
--data-dir ./data/dynerf \
--seq cut_roasted_beef \
--output-dir ./data/dynerf \
--target-points 200000 \
--width 640 \
--height 360