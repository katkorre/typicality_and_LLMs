#!/bin/bash

#SBATCH --job-name=MyExperiment

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --gres=gpu:1           # Request 1 GPU

#SBATCH --partition=l40s       # Specify GPU partition, find idle ones with sinfo

#SBATCH --output=run_%N_%j.log

#SBATCH --chdir=/scratch.hpc/username/


export APPTAINER_CACHEDIR=/scratch.hpc/username/

# Make sure to create these folders
export HF_HUB_CACHE=/scratch.hpc/username/hf_hub_cache
export HF_HOME=/scratch.hpc/username/hf_home
export NLTK_DATA=/scratch.hpc/username/nltk_data


echo "Starting execution..."

# Use --nv for NVIDIA GPUs (or --rocm for AMD GPUs)

apptainer run --nv env.sif runnables/scattergories.py --args

echo "Execution finished!"