#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0134
#PBS -v RTYPE=rt_HF
#PBS -l select=8:ncpus=8:mpiprocs=8:ngpus=8
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/optimal-sparsity-math-d1024-E32-k4-3.5B-A670M
#PBS -m n

cd $PBS_O_WORKDIR

EXP_NAME="optimal-sparsity-math-d1024-E32-k4-3.5B-A670M"

set -eu -o pipefail

# Setup environment
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1
module load cudnn/9.5/9.5.1
module load hpcx/2.20
module load nccl/2.23/2.23.4-1

source venv/bin/activate

JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

NUM_NODES=$(sort -u $PBS_NODEFILE | wc -l)
NUM_GPUS_PER_NODE=8

## Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

# model config
HIDDEN_SIZE=1024
FFN_HIDDEN_SIZE=2048
NUM_LAYERS=16
NUM_HEADS=8
NUM_QUERY_GROUPS=8
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE} * ${EXPERT_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024

LR=4e-4
MIN_LR=4e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# total number of iterations
# 125,000,000,000 (number of tokens) / 4096 (seq len) / 1024 (batch size) = 29802.322387695312 -> 29803
LR_WARMUP_STEPS=2000
LR_DECAY_ITERS=29803
TRAIN_STEPS=29803

CACHE_DIR=dclm_dedup_pes2o_math_250b_125b

# model config
TOKENIZER_MODEL=src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_LOAD_DIR=checkpoints/megatron/${EXP_NAME}
CHECKPOINT_SAVE_DIR=checkpoints/megatron/${EXP_NAME}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

TRAIN_DATA_PATH=""

TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354118153 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_01_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354696624 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_01_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354096473 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_01_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354499207 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_01_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1338045741 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_01_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1353503061 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_02_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1345257721 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_02_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1355042378 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_02_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354704854 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_02_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1349603053 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_02_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1349195514 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_03_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1360038220 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_03_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1364781862 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_03_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1339863291 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_03_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1347839889 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_03_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356559120 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_04_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1350803633 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_04_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1362158144 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_04_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1351352848 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_04_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1344176519 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_04_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356539971 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_05_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1365222266 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_05_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1348208892 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_05_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1353823425 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_05_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1346804082 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_05_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1347506794 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_06_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354977451 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_06_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356963321 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_06_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1353314605 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_06_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1350555115 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_06_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1362855670 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_07_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1334053889 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_07_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1355200079 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_07_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1353817003 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_07_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356302125 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_07_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1367117394 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_08_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1344913588 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_08_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1351196293 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_08_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356572897 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_08_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1349833067 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_08_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1341400857 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_09_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356263707 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_09_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1354844037 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_09_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1351741404 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_09_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1348634986 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_09_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1348399499 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_10_of_10_part1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356459777 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_10_of_10_part2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1337545871 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_10_of_10_part3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1363575652 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_10_of_10_part4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1350878420 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/global-shard_10_of_10_part5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2174159 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_codesearchnet-owmfilter_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2841494 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_gsm8k_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 29749546 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/WebInstructFull_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 31677007 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_dolmino_math_synth_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 67704517 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/orca-math-word-problems-200k_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 85423408 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_metamath-owmfilter_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 250390697 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_tulu_math_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 340246694 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/OpenMathInstruct-1_correct_solutions_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 446376544 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/NuminaMath-CoT_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1022226926 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/natural_reasoning_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1069981785 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/StackMathQA_preprocessed_stackexchange-math--1q1a_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4098243004 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_mathcoder2-synthmath_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5168587858 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/OpenMathInstruct-2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6944299886 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_tinyGSM-MIND_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18485484042 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/merged_tulu_flan_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5494262694 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_dolma-books_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 62853772802 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_dolma-pes2o_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3896965449 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_dolma-wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1464772187 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_dolmino-stackexchange_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10335599308 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_finemath-4plus_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2781710 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_gsm8k_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9176535715 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_mathpile_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13280211413 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_olmo-algebraicstack_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22219529548 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_olmo-arxiv_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13395295861 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_olmo-openwebmath_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4744259830 /groups/gcg51557/experiments/0134_moe_reasoning/corpus/tokenized/en/en_wiki_0000_text_document"

WANDB_ENTITY="llm-jp"
WANDB_PROJECT="0134_moe"
WANDB_NAME="${EXP_NAME}"


# Model arguments
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_HEADS}
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --num-query-groups ${NUM_QUERY_GROUPS}
    --no-masked-softmax-fusion
    --rotary-base 10000
)

MOE_ARGS=(
    --num-experts 32
    --moe-router-topk 4
    --moe-z-loss-coeff 1e-3
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $TRAIN_DATA_PATH
    --data-cache-path $CACHE_DIR
    --split 990,10,0
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --train-iters ${TRAIN_STEPS}
    --lr-decay-iters ${TRAIN_STEPS}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --weight-decay ${WEIGHT_DECAY}
    --lr-warmup-iters ${LR_WARMUP_STEPS}
    --clip-grad ${GRAD_CLIP}
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --bf16
    --use-flash-attn
    --transformer-impl "transformer_engine"
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --distributed-backend nccl
    --ckpt-format torch
)

# Model parameters
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}
    --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE}
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --moe-per-layer-logging
    --save-interval 500
    --eval-interval 500
    --eval-iters 10
    --save ${CHECKPOINT_SAVE_DIR}
    --load ${CHECKPOINT_LOAD_DIR}
    --use-mpi
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_NAME}
    --wandb-entity ${WANDB_ENTITY}
)


export NVTE_FUSED_ATTN=0
mpirun \
    -np $NUM_GPUS \
    --hostfile $PBS_NODEFILE \
    --npernode $NUM_GPUS_PER_NODE \
    --map-by slot \
    --bind-to none \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NUM_NODES=$NUM_NODES \
    -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -x PATH \
    -x LD_LIBRARY_PATH \
    python src/Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"