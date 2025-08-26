#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0177_eval
#PBS -v RTYPE=rt_HG
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -k oed
#PBS -V
#PBS -o outputs/
#PBS -m n

# cd $PBS_O_WORKDIR
cd /home/ach17887oc/experiments/0177_MoE_reasoning_eval/large_language_monkeys
set -eu -o pipefail

export CUDA_VISIBLE_DEVICES=0


source .venv_monkey_except/bin/activate
export NUM_NODES=1
export NUM_GPU_PER_NODE=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

# MODEL=${1-:"MODEL"}
echo "MODEL: ${MODEL}"
SAVE_DIR="generated_samples"
mkdir -p ${SAVE_DIR}
SAVE_DIR="${SAVE_DIR}/${MODEL}"
mkdir -p ${SAVE_DIR}

RESULT_DIR="results"
mkdir -p ${RESULT_DIR}
RESULT_DIR="${RESULT_DIR}/${MODEL}"
mkdir -p ${RESULT_DIR}

export NUM_FEW_SHOT=0
export TEMPERATURE=0.0

# TRY="2"
# export GSM8K_SAVE_DIR="${SAVE_DIR}/gsm8k_${NUM_FEW_SHOT}shot_try${TRY}"
# export GSM8K_RESULT_DIR="${RESULT_DIR}/gsm8k_${NUM_FEW_SHOT}shot_try${TRY}"
export GSM8K_SAVE_DIR="${SAVE_DIR}/gsm8k_${NUM_FEW_SHOT}shot_temp${TEMPERATURE}"
export GSM8K_RESULT_DIR="${RESULT_DIR}/gsm8k_${NUM_FEW_SHOT}shot_temp${TEMPERATURE}"

# generate
python llmonk/generate/gsm8k_local.py \
  model=${MODEL} \
  save_dir="${GSM8K_SAVE_DIR}" \
  num_few_shot=${NUM_FEW_SHOT} \
  temperature=${TEMPERATURE} \
  num_samples=1 \
  batch_size=1 \
  tensor_parallel_size=${NUM_GPUS} \
  --list vllm_args --disable-log-requests list-- \
  --list stop_strings Q: Question: list--


#evaluate
python llmonk/evaluate/rewarding/gsm8k_math.py \
  dset=gsm8k \
  samples_dir="${GSM8K_SAVE_DIR}" \
  save_dir="${GSM8K_RESULT_DIR}" \
  rmp="Armo"

python llmonk/evaluate/extract_check/gsm8k.py \
  samples_dir="${GSM8K_SAVE_DIR}" \
  save_dir="${GSM8K_RESULT_DIR}"


# evaluate grpo style
export GSM8K_RESULT_DIR="${GSM8K_RESULT_DIR}_grpo"
python llmonk/evaluate/rewarding/gsm8k_grpo_eval.py \
  samples_dir="${GSM8K_SAVE_DIR}" \
  save_dir="${GSM8K_RESULT_DIR}" \
  rmp="Armo"

