for k in {1..500}
do
    echo -e "\nStarting job $k"
    sbatch run_job.sh $k
done