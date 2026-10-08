#!/bin/bash
#SBATCH --job-name=typicality-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=sbuild
#SBATCH --output=logs/build_%j.log

# Builds the image, checks the registry, and downloads both Hugging Face
# models. `apptainer` exists only on compute nodes, so all three run here and
# not on a login node. Submit from the directory that holds this project:
#
#   mkdir -p logs && sbatch cluster/build.sbatch
#
# Both models are gated. Accept their licences on the Hugging Face website,
# then store a token once with `hf auth login`, run with
# HF_HOME=$SCRATCH/hf_home so the token lands where this job reads it.

set -euo pipefail

SCRATCH=${SCRATCH:-/scratch.hpc/${USER:-$(id -un)}}

# Remove stale Apptainer layers and abandoned build directories. Do not run
# this job alongside another image build that uses the same scratch path.
echo "clearing Apptainer cache under $SCRATCH"
rm -rf -- "$SCRATCH/apptainer_cache" "$SCRATCH/apptainer_tmp"

# Home has a small quota, so the models end on scratch. Scratch is slow under
# small writes, and both downloads write in small pieces, so each goes to a
# per-job directory on the node's own disk when that disk has room for it. The
# finished models are copied to scratch in one pass. A stage without room
# falls back to scratch, where slow beats running out of room.
LOCAL="${TMPDIR:-/tmp}/${USER:-$(id -un)}-typicality-build-${SLURM_JOB_ID:-$$}"
SCRATCH_TMP="$SCRATCH/apptainer_tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL" "$SCRATCH_TMP"' EXIT

free_gb() {
  df -Pk "$LOCAL" | awk 'NR==2 { print int($4 / 1048576) }'
}

# The layers take about 4 GB and the unpacked image about 10 GB.
echo "local disk has $(free_gb) GB free"
if [ "$(free_gb)" -ge 20 ]; then
  export APPTAINER_CACHEDIR="$LOCAL/apptainer_cache"
  export APPTAINER_TMPDIR="$LOCAL/apptainer_tmp"
else
  echo "under 20 GB, building on scratch instead (slow)" >&2
  export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
  export APPTAINER_TMPDIR="$SCRATCH_TMP"
fi
export HF_HOME="$SCRATCH/hf_home"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

# `--ignore-fakeroot-command` is required. The user is not in /etc/subuid, so
# apptainer injects a host `libfakeroot.so` built against GLIBC 2.38 into an
# Ubuntu 22.04 image with 2.35, and `%post` dies before its first line.
echo "building env.sif"
apptainer build --ignore-fakeroot-command --force env.sif cluster/env.def

echo "checking the registry"
apptainer exec --bind "$SCRATCH" --env PYTHONPATH="$PWD" env.sif python -c "
from pathlib import Path
from cinnamon.registry import Registry
valid, invalid = Registry.build(directory=Path('.').resolve())
print(len(valid), 'valid,', len(invalid), 'invalid')
assert not invalid, sorted(str(key) for key in invalid)"

# Frees the local disk for the models.
rm -rf "$LOCAL/apptainer_cache" "$LOCAL/apptainer_tmp"

# About 16 GB for Llama and 24 GB for Gemma. The patterns skip Llama's
# `original/` directory, a second copy of the weights that transformers never
# reads.
echo "local disk has $(free_gb) GB free"
if [ "$(free_gb)" -ge 50 ]; then
  DOWNLOAD="$LOCAL/hf_home"
  mkdir -p "$DOWNLOAD"
  if [ -f "$HF_HOME/token" ]; then cp "$HF_HOME/token" "$DOWNLOAD/token"; fi
else
  echo "under 50 GB, downloading the models on scratch instead (slow)" >&2
  DOWNLOAD="$HF_HOME"
fi
echo "downloading the models"
apptainer exec --bind "$SCRATCH" --bind "$LOCAL" --env HF_HOME="$DOWNLOAD" \
  env.sif python -c "
from huggingface_hub import snapshot_download
for repo in ('meta-llama/Llama-3.1-8B-Instruct', 'google/gemma-3-12b-it'):
    print(snapshot_download(repo, allow_patterns=['*.json', 'tokenizer*', 'model-*.safetensors']))"
# `-a` keeps the cache's symlinks and hard links.
if [ "$DOWNLOAD" != "$HF_HOME" ]; then
  echo "copying the models to scratch"
  cp -a "$DOWNLOAD/." "$HF_HOME/"
fi

echo "done"
