#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 05-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o traceback/sgd.out


python -u main.py --name=sgd_exp --train=True --SGD=True --device=cuda --batch_size=512 > outputs/sgd.out
