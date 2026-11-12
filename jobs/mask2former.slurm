#!/bin/bash
# Score the pretrained Mask2Former on our 10 clue classes (reference row, no training).
# Submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/mask2former.slurm
# The results are also saved to /fp/projects01/ec674/williaan/results/mask2former.txt

#SBATCH --job-name=seg-m2f
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8          # one per data loader worker
#SBATCH --mem=32G                  # Swin-L is much bigger than DINOv2-small
#SBATCH --time=01:00:00            # predictions only, but a bigger model on 2x larger images
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit
# Without this, a crash in python is hidden because the job only sees tee succeed.
set -o pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# Mask2Former was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

RESULTS=/fp/projects01/ec674/williaan/results
mkdir -p $RESULTS

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
nvidia-smi
# tee prints to the Slurm log and saves a copy under a readable name.
python mask2former_eval.py "$@" | tee $RESULTS/mask2former.txt
