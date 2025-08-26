#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -N 0149_verl
#PBS -o outputs/
#PBS -j oe
#PBS -v RTYPE=rt_HF

set -eu

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
source ~/.bash_profile

module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2

pip install -U pip wheel packaging ninja
pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install vllm==0.6.3
