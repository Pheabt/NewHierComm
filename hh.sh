#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=partition_name
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd /home/iotsc_g4/aaa/NewHierComm
conda activate tie
python main.py --agent ac_att --att_head 2
