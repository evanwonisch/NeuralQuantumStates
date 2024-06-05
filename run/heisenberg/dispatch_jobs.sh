# !/bin/bash

for k in $(seq 1 400)
do
    for seed in $(seq 1 10) 
    do
        sbatch run_job.sh $k $seed
    done
done