# !/bin/bash

for k in $(seq 0 224)
do
    for seed in $(seq 0 9) 
    do
        sbatch run_job.sh $k $seed
    done
done