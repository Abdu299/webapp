#!/bin/bash
# Train the linear decoder on one GPU on Fox.
# Submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/train_linear.slurm                 # seed 0, default settings
#     sbatch jobs/train_linear.slurm --seed 1        # extra arguments go to train.py

#SBATCH --job-name=seg-linear
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8          # one per data loader worker
#SBATCH --mem=32G
#SBATCH --time=02:00:00            # the job is stopped after this
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
