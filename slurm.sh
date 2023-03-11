#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=quafel
# 
# 
#SBATCH --ntasks=200
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=40:00:00
# 
# partition the job will run on
#SBATCH --partition multiple
# 
# expected memory requirements
#SBATCH --mem=80000MB

kedro run --pipeline measure --runner ParallelRunner --async

# Done
exit 0
