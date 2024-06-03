# !/bin/bash

for k in {1..3}
do
    echo -e "\nStarting job $k"
    sbatch run_job.sh $k
done