#!/bin/bash

MODEL_NAME_PATH=llm-jp/optimal-sparsity-math-d512-E8-k2-320M-A170M
GENERAL_TASK_NAME="triviaqa,gsm8k,hellaswag"
GENERAL_NUM_FEWSHOT=4
GENERAL_NUM_TESTCASE="all"
GENERAL_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${GENERAL_NUM_FEWSHOT}shot_${GENERAL_NUM_TESTCASE}cases/general"
TP=1
DP=1

mkdir -p $GENERAL_OUTDIR

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,max_length=4096,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $GENERAL_TASK_NAME \
    --num_fewshot $GENERAL_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "$GENERAL_OUTDIR" \
    --use_cache "$GENERAL_OUTDIR" \
    --log_samples \
    --seed 42

GSM8K_TASK_NAME="gsm_plus"
GSM8K_NUM_FEWSHOT=5
GSM8K_NUM_TESTCASE="gsm_plus"
GSM8K_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${GSM8K_NUM_FEWSHOT}shot_${GSM8K_NUM_TESTCASE}cases/gsm8k"

mkdir -p $GSM8K_OUTDIR

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $GSM8K_TASK_NAME \
    --num_fewshot $GSM8K_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "$GSM8K_OUTDIR" \
    --use_cache "$GSM8K_OUTDIR" \
    --log_samples \
    --seed 42
