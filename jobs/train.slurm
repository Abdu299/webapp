#!/bin/bash
# Train any decoder on one GPU on Fox. All arguments go to train.py.
# Week 2 runs, submitted from the repo root (they run at the same time):
#     mkdir -p logs
#     sbatch jobs/train.slurm --model linear --class-weights
#     sbatch jobs/train.slurm --model segmenter --epochs 10 --lr 1e-4
#     sbatch jobs/train.slurm --model segmenter --class-weights --epochs 10 --lr 1e-4
# Higher resolution (48x36 patches instead of 32x24, for thin classes like poles):
#     sbatch jobs/train.slurm --model segmenter --size 672 504 --epochs 10 --lr 1e-4
# Segmenter gets lr 1e-4: its transformer layers are much bigger than one linear
# layer, and 1e-3 with AdamW often makes transformers unstable.
# It gets 10 epochs because it has ~4.0M parameters to learn instead of 3,850.

#SBATCH --job-name=seg-train
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8          # one per data loader worker
#SBATCH --mem=32G
#SBATCH --time=05:00:00            # 10 Segmenter epochs took ~1 h at 448x336; 672x504 has ~2.25x the patches
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# DINOv2 was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
nvidia-smi
python train.py "$@"
