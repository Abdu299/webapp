#!/bin/bash
# Country accuracy of Abdu's GeoVit on a test set, without blur (week 4 sanity check).
# Compares his input (whole photo squashed) with ours (448 centre square).
# Give the test set name, submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/geovit_check.slurm osv5m
# Output: logs/seg-geovit-<job id>.out and /fp/projects01/ec674/williaan/results/geovit_check_<set>.txt

#SBATCH --job-name=seg-geovit
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                  # DINOv2-small is small
#SBATCH --time=00:30:00            # 2 guesses per photo, a few minutes for 1,233 photos
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit
# Without this, a crash in python is hidden because the job only sees tee succeed.
set -o pipefail

# Which test set: im2gps or osv5m
SET=$1

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# DINOv2 was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
python geovit_check.py --csv /fp/projects01/ec674/williaan/data/test_sets/$SET.csv "${@:2}" \
    | tee /fp/projects01/ec674/williaan/results/geovit_check_$SET.txt
