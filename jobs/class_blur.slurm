#!/bin/bash
# Blur one clue class at a time and record the OSV-5M baseline's guesses (week 3).
# Needs the masks from predict.slurm. Give the test set name, submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/class_blur.slurm im2gps
#     sbatch jobs/class_blur.slurm osv5m
# Extra arguments go to class_blur.py, e.g. --limit 4 for a quick test.
# Abdu's model instead of the OSV-5M baseline (sbatch passes MODEL on to the job):
#     MODEL=geovit sbatch jobs/class_blur.slurm osv5m
# Output: /fp/projects01/ec674/williaan/results/class_blur_<set>.csv (osv5m model)
#     or .../class_blur_geovit_<set>.csv (geovit). Rank it with clue_ranking.py

#SBATCH --job-name=seg-blur
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # the baseline is a CLIP ViT-L, much bigger than DINOv2-small
#SBATCH --time=00:45:00            # 21 guesses per photo, ~1,200 photos for OSV-5M
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit

# Which test set: im2gps or osv5m
SET=$1
# Which model (default: the OSV-5M baseline). File names stay the same as before for osv5m.
MODEL=${MODEL:-osv5m}
NAME=""; if [ "$MODEL" != osv5m ]; then NAME="${MODEL}_"; fi

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# The OSV-5M baseline was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
nvidia-smi
python class_blur.py --model $MODEL --csv /fp/projects01/ec674/williaan/data/test_sets/$SET.csv \
    --masks /fp/projects01/ec674/williaan/masks/$SET \
    --out /fp/projects01/ec674/williaan/results/class_blur_$NAME$SET.csv "${@:2}"
