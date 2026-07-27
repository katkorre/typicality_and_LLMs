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
# Replace python-repo with your python repository
export PYTHONPATH="$PYTHONPATH:/scratch.hpc/username/python-repo"
export HF_HUB_CACHE=/scratch.hpc/username/hf_hub_cache
export HF_HOME=/scratch.hpc/username/hf_home
export NLTK_DATA=/scratch.hpc/username/nltk_data


echo "Starting execution..."

# Use --nv for NVIDIA GPUs (or --rocm for AMD GPUs)

apptainer run --nv my_env.sif llmprototypes/runnables/scattergories.py --model gemma --language english --strategy 0 --letters c -t HF-KEY


echo "Execution finished!"