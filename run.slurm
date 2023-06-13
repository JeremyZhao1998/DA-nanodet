#!/bin/bash
#SBATCH --job-name=nanodet          # job name
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=24      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --partition=gpu
#SBATCH --qos=level4
#SBATCH --account=research
#SBATCH --time=10:00:00         # total run time limit (HH:MM:SS)
#SBATCH --output=output/city2foggy/source_only/log.txt

module purge
module load slurm/slurm/20.11.8
module load anaconda3/2021.11
source activate
conda activate at

python tools/train.py configs/tencent_sunny2heavy.yml
