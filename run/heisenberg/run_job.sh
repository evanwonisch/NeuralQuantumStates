#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=my_log.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_dist
#SBATCH --account=ndqm

source ../../.venv/bin/activate

echo -e "Starting job" $1

python -m run_momentum $1