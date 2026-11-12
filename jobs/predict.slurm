#!/bin/bash
# Predict clue masks for both week 3 test sets with the best segmenter (segmenter_896_seed0).
# Needs the test set CSVs from test_sets.py. Submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/predict.slurm
# Masks go to /fp/projects01/ec674/williaan/masks/{im2gps,osv5m}/, with previews in */preview/

#SBATCH --job-name=seg-predict
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00            # ~1,400 photos, one at a time, no training
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit
# Without this, a crash in python is hidden because the job only sees tee succeed.
set -o pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# DINOv2 was already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

DATA=/fp/projects01/ec674/williaan/data/test_sets
MASKS=/fp/projects01/ec674/williaan/masks
RESULTS=/fp/projects01/ec674/williaan/results
mkdir -p $RESULTS

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
for SET in osv5m im2gps; do
    # Skip a test set that is not made yet (Im2GPS needs Victoria's photos)
    if [ -f $DATA/$SET.csv ]; then
        # tee keeps the class shares per test set, to compare with Vistas
        python predict.py --csv $DATA/$SET.csv --out $MASKS/$SET "$@" | tee $RESULTS/predict_$SET.txt
    fi
done
