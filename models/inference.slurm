#!/bin/bash
#SBATCH --job-name=smollrx-135M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpua100
#SBATCH --mem=24G
#SBATCH --output=/ai_studio/logs/inference/inf-%j.out
#SBATCH --error=/ai_studio/logs/inference/inf-%j.err
#SBATCH --time=23:59:59

set -e
module purge
module load anaconda3/2020.02/gcc-9.2.0
source activate slm_training

CKPT_PATH="/trained_models/135M/final"
nvidia-smi

# Show some environment variables
echo python3 version = `python3 --version`
echo "Python path: $(which python3)"
echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"

# List currently loaded modules for debugging
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

torchrun --nproc_per_node=1 /nanotron/run_generate.py --ckpt-path $CKPT_PATH

echo "END TIME: $(date)"