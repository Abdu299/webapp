#!/bin/bash
# Evaluate one trained decoder on the Vistas validation set (per-class IoU).
# Submit from the repo root, giving the checkpoint name without .pt:
#     mkdir -p logs
#     sbatch jobs/evaluate.slurm linear_seed0
#     sbatch jobs/evaluate.slurm segmenter_weighted_seed0
#     sbatch jobs/evaluate.slurm segmenter_672_seed0 --size 672 504   (extra arguments go to evaluate.py)
# The results are also saved to /fp/projects01/ec674/williaan/results/<name>.txt

#SBATCH --job-name=seg-eval
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8          # one per data loader worker
#SBATCH --mem=16G                  # training peaked at 4.3G, so 32G was far more than needed
#SBATCH --time=00:30:00            # one pass over 2000 images, no training
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit
# Without this, a crash in python is hidden because the job only sees tee succeed.
set -o pipefail

# Which checkpoint to evaluate, e.g. linear_seed0 or segmenter_weighted_seed0
NAME=$1
# The model is the part of the name before the first "_" (train.py names them this way).
MODEL=${NAME%%_*}

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# DINOv2 was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

RESULTS=/fp/projects01/ec674/williaan/results
mkdir -p $RESULTS

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
# tee prints to the Slurm log and saves a copy under a readable name.
python evaluate.py --model $MODEL --checkpoint /fp/projects01/ec674/williaan/checkpoints/$NAME.pt "${@:2}" | tee $RESULTS/$NAME.txt
