#!/usr/bin/bash -l
#SBATCH --job-name=schY3
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=[0-4]%2
#SBATCH --mem=64GB
#SBATCH --time=18:00:00
#SBATCH --nodes=1


source self/bin/activate
python /home/ygence/schechter.py -r $SLURM_ARRAY_TASK_ID