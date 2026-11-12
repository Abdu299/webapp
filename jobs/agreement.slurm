#!/bin/bash
# Check that our segmenter still works on the week 3 photos: compare it with Mask2Former
# on Vistas validation photos (the control) and on both test sets.
# Needs the test set CSVs from test_sets.py. Submit from the repo root:
#     mkdir -p logs
#     sbatch jobs/agreement.slurm
# Results: /fp/projects01/ec674/williaan/results/agreement_{vistas,osv5m,im2gps}.txt

#SBATCH --job-name=seg-agree
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4          # one per data loader worker
#SBATCH --mem=32G                  # Swin-L is much bigger than DINOv2-small
#SBATCH --time=01:00:00            # ~2,000 photos through Mask2Former, like the week 2 job
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id

# Stop at the first error instead of running the rest of the script.
set -o errexit
# Without this, a crash in python is hidden because the job only sees tee succeed.
set -o pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /fp/projects01/ec674/williaan/venv/bin/activate

# DINOv2 and Mask2Former were already downloaded to this cache on the login node.
# Offline mode stops it from trying the internet from the GPU node.
export HF_HOME=/fp/projects01/ec674/williaan/hf_cache
export HF_HUB_OFFLINE=1

DATA=/fp/projects01/ec674/williaan/data/test_sets
RESULTS=/fp/projects01/ec674/williaan/results
mkdir -p $RESULTS

cd /fp/projects01/ec674/williaan/g02-p3/segmenter
nvidia-smi
# 500 Vistas photos are enough for the control and save GPU time
python agreement.py vistas --limit 500 | tee $RESULTS/agreement_vistas.txt
for SET in osv5m im2gps; do
    if [ -f $DATA/$SET.csv ]; then
        python agreement.py $DATA/$SET.csv | tee $RESULTS/agreement_$SET.txt
    fi
done
