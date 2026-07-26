#!/bin/bash

#SBATCH --job-name=BuildEnv

#SBATCH --ntasks=1

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=16G

#SBATCH --partition=sbuild

#SBATCH --output=build_%N_%j.log

#SBATCH --chdir=/scratch.hpc/username/

# Force Apptainer to use scratch for caching to avoid filling up the home directory

export APPTAINER_CACHEDIR=/scratch.hpc/username/

echo "Building container..."

apptainer build env.sif env.def

echo "Done!"