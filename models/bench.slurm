#!/bin/bash
#SBATCH --job-name=bench-smollrx-135M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpua100
#SBATCH --mem=24G
#SBATCH --output=/ai_studio/logs/bench/bench-%j.out
#SBATCH --error=/ai_studio/logs/bench/bench-%j.err
#SBATCH --time=00:59:59

set -e
module purge
module load anaconda3/2020.02/gcc-9.2.0
source activate slm_training

MAIN_SCRIPT="/.conda/envs/slm_training/lib/python3.10/site-packages/lighteval/__main__.py"
CHECKPOINT_CONFIG="/trained_models/smolrx_135M/PM03A07FW90-FPT/35000/config.yaml"
LIGHTEVAL_CONFIG="/ai_studio/configs/lighteval_config.yaml"
echo "Checkpoint Path: $CHECKPOINT_CONFIG"
nvidia-smi

export TRANSFORMERS_CACHE="/ai_studio/cache"
export HF_HOME="/ai_studio/cache"

# Show some environment variables
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
export HF_HUB_TOKEN=<key>

echo "Number of nodes: $COUNT_NODE"
echo "Hostnames: $HOSTNAMES"

# Display some environment information.
echo "python3 version = $(python3 --version)"
echo "Python path: $(which python3)"
echo "NCCL version: $(python -c 'import torch; print(torch.cuda.nccl.version())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"

# Launch the torchrun command using the defined variables.
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  $MAIN_SCRIPT nanotron \
  --checkpoint-config-path $CHECKPOINT_CONFIG \
  --lighteval-config-path $LIGHTEVAL_CONFIG

echo "END TIME: $(date)"