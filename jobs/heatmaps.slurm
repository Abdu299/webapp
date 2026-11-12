#!/bin/bash
# Occlusion heatmaps of the OSV-5M baseline for every photo of a test set (week 4).
# Does not need masks: the heatmaps are compared with the masks afterwards (heatmap_ranking.py).
# Give the test set name, submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/heatmaps.slurm osv5m
# Extra arguments go to heatmaps.py, e.g. --limit 4 for a quick test.
# Abdu's model instead of the OSV-5M baseline (sbatch passes MODEL on to the job):
#     MODEL=geovit sbatch jobs/heatmaps.slurm osv5m
# Output: /fp/projects01/ec674/williaan/heatmaps/<set>/<name>.npy, one [448, 448] float16 heatmap (km) per photo
#     geovit: .../heatmaps_geovit/<set>/<name>.npy (drop in probability)

#SBATCH --job-name=seg-heat
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # the baseline is a CLIP ViT-L, much bigger than DINOv2-small
#SBATCH --time=03:00:00            # 729 guesses per photo, ~6 s per photo: ~2 h for 1,233 OSV-5M photos
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit

# Which test set: im2gps or osv5m
SET=$1
# Which model (default: the OSV-5M baseline). Folder names stay the same as before for osv5m.
MODEL=${MODEL:-osv5m}
FOLDER=""; if [ "$MODEL" != osv5m ]; then FOLDER="_$MODEL"; fi

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# The OSV-5M baseline was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
nvidia-smi
python heatmaps.py --model $MODEL --csv /fp/projects01/ec674/williaan/data/test_sets/$SET.csv \
    --out /fp/projects01/ec674/williaan/heatmaps$FOLDER/$SET "${@:2}"
