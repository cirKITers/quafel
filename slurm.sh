#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=quafel
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=80
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=4:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=64000MB

kedro run --pipeline parallel --runner ParallelRunner --async

# Done
exit 0
