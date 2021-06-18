#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_shared_jupyter
#SBATCH --gpus=gtx1080ti:2
#SBATCH --gpus-per-node=gtx1080ti:2
#SBATCH -c 6
#SBATCH -t 3:00
#SBATCH -o gpipe-answer-%j.out
#SBATCH -e gpipe-answer-%j.out
#SBATCH --reservation=prace_ml_course

source ${TEACHER_DIR}/JHL_hooks/env

tar -C "$TMPDIR" -zxf ${TEACHER_DIR}/JHL_data/MNIST.tar.gz

export PYTHONUNBUFFERED=1
python mnist_gpipe_answer.py --arch resnet50 --datadir="$TMPDIR" --batchsize=512 --num_microbatches=6 --balance_by=time
