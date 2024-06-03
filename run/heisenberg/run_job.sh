# !/bin/bash

# SBATCH --job-name=my_job
# SBATCH --output=my_log.log

# SBATCH --ntasks=1
# SBATCH --cpus-per-task=4

# SBATCH --time=100:00:00
# SBATCH --partition=cpu_shared
# SBATCH --account=ndqm

module purge

module load anaconda3

source .venv/bin/activate

echo -e "Starting job" $1

python -m run_momentum $1