#!/bin/bash

MODEL_NAME_PATH=llm-jp/optimal-sparsity-code-d512-E8-k2-320M-A170M

MBPP_TASK_NAME="mbpp,mbpp_plus"
MBPP_NUM_FEWSHOT=3
MBPP_NUM_TESTCASE="all"
MBPP_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${MBPP_NUM_FEWSHOT}shot_${MBPP_NUM_TESTCASE}cases/mbpp"

HUMANEVAL_TASK_NAME="humaneval,humaneval_plus"
HUMANEVAL_NUM_FEWSHOT=0
HUMANEVAL_NUM_TESTCASE="all"
HUMANEVAL_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${HUMANEVAL_NUM_FEWSHOT}shot_${HUMANEVAL_NUM_TESTCASE}cases/humaneval"
TP=1
DP=1

mkdir -p $MBPP_OUTDIR
mkdir -p $HUMANEVAL_OUTDIR
export HF_ALLOW_CODE_EVAL="1"

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,max_length=4096,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $HUMANEVAL_TASK_NAME \
    --num_fewshot $HUMANEVAL_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "$HUMANEVAL_OUTDIR" \
    --use_cache "$HUMANEVAL_OUTDIR" \
    --log_samples \
    --confirm_run_unsafe_code \
    --seed 42


lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,max_length=4096,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $MBPP_TASK_NAME \
    --num_fewshot $MBPP_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "$MBPP_OUTDIR" \
    --use_cache "$MBPP_OUTDIR" \
    --log_samples \
    --confirm_run_unsafe_code \
    --seed 42