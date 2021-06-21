#!/bin/bash
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH -c 3
#SBATCH -t 0:10:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:gtx1080ti:2
#SBATCH --gpus-per-node=gtx1080ti:2
#SBATCH --reservation=prace_ml_course
#SBATCH -o gpu_2_%A.out

echo "Performing Training..."


NODE_LIST=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:1,/g' )
NODE_LIST=${NODE_LIST%?}
echo $NODE_LIST

time horovodrun -np 2 --hosts $NODE_LIST python cifar10_horovod.py

