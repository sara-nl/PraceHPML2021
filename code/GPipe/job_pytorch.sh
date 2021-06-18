#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_shared_jupyter
#SBATCH --gpus=gtx1080ti:1
#SBATCH -c 3
#SBATCH -t 3:00
#SBATCH -o pytorch-%j.out
#SBATCH -e pytorch-%j.out
#SBATCH --reservation=prace_ml_course

module purge

source ${TEACHER_DIR}/JHL_hooks/env

tar -C "$TMPDIR" -zxf ${TEACHER_DIR}/JHL_data/MNIST.tar.gz

export PYTHONUNBUFFERED=1
python mnist_pytorch.py --arch resnet50 --datadir="$TMPDIR" --batchsize=256
