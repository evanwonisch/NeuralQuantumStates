for k in {1..400}
do
    for seed in {1,10}
    do
        sbatch run_job.sh $k $seed
    done
done