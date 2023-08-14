#!/bin/bash -x

#SBATCH --job-name=irsde-rician-50

#SBATCH --account=delia-mp

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

#####################################################################################
# TODO: need to modify every time, and need to manually create a directory named "<date>_<number_of_experiment>"
#SBATCH --error=/p/scratch/delia-mp/lin4/experiment_result/irsde/0804-rician-50/error-%j.err
#SBATCH --output=/p/scratch/delia-mp/lin4/experiment_result/irsde/0804-rician-50/output-%j.out

#####################################################################################

#SBATCH --time=24:00:00
#SBATCH --partition=dc-gpu
#SBTACH --gres=gpu:4

# *** start of job script ***
# Note: The current working directory at this point is the directory where sbatch was executed.

echo 'Experiment start !'

# Without this, srun does not inherit cpus-per-task from sbatch.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
# export MASTER_PORT=7010

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPU_PER_NODE=4


CONFIG=codes/config/deraining/options/train/ir-sde-rician-50.yml

source /p/project/delia-mp/lin4/Julich_experiment/image-restoration-sde/irsde_env/activate.sh

srun python3 -m \
    torch.distributed.launch \
    --nproc_per_node=4 \
    codes/config/deraining/train.py \
    -opt $CONFIG \
    --launcher pytorch

# srun torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPU_PER_NODE \
#     train.py \
#     -opt=options/train/ir-sde-multi.yml \
#     --launcher pytorch \

echo 'Experiment End !'