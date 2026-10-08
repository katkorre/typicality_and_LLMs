#!/bin/bash
#SBATCH --job-name=typicality
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

# Plays every game of one Hugging Face model into results/scattergories.
# Submit from the directory that holds this project, after
# cluster/build.sbatch:
#
#   sbatch cluster/run.sbatch llama --smoke   # 3 categories into results/smoke
#   sbatch cluster/run.sbatch llama           # every game
#   sbatch cluster/run.sbatch gemma --language german
#
# A finished game is skipped, so a job that hits its time limit continues
# from the unfinished game when it is submitted again. The API models, gpt
# and claude, need no GPU and run outside the cluster.
#
# The partition has to be named: the default is `sbuild`, which has no GPU.

set -euo pipefail

MODEL=${1:?name a model: llama or gemma}
case "$MODEL" in
  llama|gemma) ;;
  *) echo "the cluster runs llama or gemma, not '$MODEL'" >&2; exit 2 ;;
esac

SCRATCH=${SCRATCH:-/scratch.hpc/${USER:-$(id -un)}}
mkdir -p logs results

if [ ! -f env.sif ]; then
  echo "missing $PWD/env.sif, run cluster/build.sbatch first" >&2
  exit 2
fi

echo "$(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Offline, so a model that build.sbatch did not download fails at once rather
# than downloading on a node that may have no network.
apptainer exec --nv --bind "$SCRATCH" \
  --env PYTHONPATH="$PWD" \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HOME="$SCRATCH/hf_home" \
  --env HF_HUB_OFFLINE=1 \
  env.sif python run.py "$@"

echo "$(date -Is) done"
