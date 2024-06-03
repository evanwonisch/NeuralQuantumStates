for k in {1..5}
do
    echo -e "\nStarting job $k"
    sbatch run_job.sh $k
done