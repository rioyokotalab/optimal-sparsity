#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -N 0149_verl
#PBS -o outputs/log.txt
#PBS -j oe
#PBS -v RTYPE=rt_HF

set -eux

cd ${PBS_O_WORKDIR}

exec >${PBS_O_WORKDIR}/outputs/${PBS_JOBID}.txt 2>&1

source /etc/profile.d/modules.sh
source ~/.bash_profile

module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "START: $start_time"

mapfile -t UNIQUE_NODES < <(awk '{print $1}' $PBS_NODEFILE)
NUM_NODES=${#UNIQUE_NODES[@]}
GPUS_PER_NODE=8

MODEL_PATH=llm-jp/optimal-sparsity-math-d1024-E16-k2-1.9B-A470M

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
python3 examples/data_preprocess/math_dataset.py --local_dir data/math

ACTOR_OPTIM_LR=5e-6
EXP_NAME="optimal-sparsity-math-d1024-E16-k2-1.9B-A470M"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="data/math/train.parquet" \
    data.val_files=["data/gsm8k/test.parquet","data/math/test.parquet"] \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_OPTIM_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=hf \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name=${EXP_NAME} \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=100000 \
    trainer.test_freq=5 \
    trainer.total_epochs=15

python scripts/model_merger.py --local_dir checkpoints/verl_grpo_example_gsm8k/${EXP_NAME}/global_step_105/actor/

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "END: $end_time"

start_sec=$(date -d "$start_time" +%s)
end_sec=$(date -d "$end_time" +%s)
elapsed_time=$((end_sec - start_sec))

echo "TOTAL: $(($elapsed_time / 3600))h $(($elapsed_time % 3600 / 60))m $(($elapsed_time % 60))s"