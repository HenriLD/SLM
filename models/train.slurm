#!/bin/bash
#SBATCH --job-name=smollrx-135M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpua100
#SBATCH --mem=24G
#SBATCH --output=/ai_studio/logs/train/train-%j.out
#SBATCH --error=/ai_studio/logs/train/train-%j.err
#SBATCH --time=23:59:59

set -e
module purge
module load anaconda3/2020.02/gcc-9.2.0
source activate slm_training

TRAINER_PYTHON_FILE="/nanotron/run_train.py"
CONFIG_PATH_YAML="/ai_studio/configs/config_smollrx_135M.yaml"
nvidia-smi


echo python3 version = `python3 --version`
echo "Python path: $(which python3)"
echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "Loaded modules:"
module list

echo "START TIME: $(date)"
secs_to_human() {
    echo "$(( ${1} / 3600 )):$(( (${1} / 60) % 60 )):$(( ${1} % 60 ))"
}
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"


export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export CUDA_DEVICE_MAX_CONNECTIONS="1"

echo "Number of nodes: $COUNT_NODE"
echo "Hostnames: $HOSTNAMES"

CMD=" $TRAINER_PYTHON_FILE \
    --config-file $CONFIG_PATH_YAML \
    "
export LAUNCHER="torchrun \
    --nproc_per_node 4 \
    --nnodes $COUNT_NODE \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

# Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

export WANDB_API_KEY= <key>

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"