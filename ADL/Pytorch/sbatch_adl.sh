#!/bin/bash -x

#SBATCH --job-name=adl-mixed-50

#SBATCH --account=delia-mp

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

#####################################################################################
# TODO: need to modify every time, and need to manually create a directory named "<date>_<number_of_experiment>"
#SBATCH --error=/p/scratch/delia-mp/lin4/experiment_result/adl/0804-mixed-50/error-%j.err
#SBATCH --output=/p/scratch/delia-mp/lin4/experiment_result/adl/0804-mixed-50/output-%j.out

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
export MASTER_PORT=7010


export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPU_PER_NODE=4

source /p/project/delia-mp/lin4/Julich_experiment/ADL/adl_env/activate.sh

export LOG_DIR="/p/scratch/delia-mp/lin4/experiment_result/adl/"
export EXPERIMENT="0804-mixed-50"
export CHANNELS_NUM=1

srun python3 train.py \
	--EXPERIMENT ${EXPERIMENT} \
    --json-file configs/ADL_train.json \
    --DENOISER efficient_Unet \
    --effective_batch_size 16 \
    --world_size 4 \
    --noise_type mixed \
    --noise_level 50 \
    --use_special_loss \
	--CHANNELS-NUM ${CHANNELS_NUM} \
    --train-dirs '/p/scratch/delia-mp/lin4/RestormerGrayScaleData/clean/train/DFWB' \
                '/p/scratch/delia-mp/lin4/RestormerGrayScaleData/clean/test/BSD68' \
    --test-dirs '/p/scratch/delia-mp/lin4/RestormerGrayScaleData/clean/test/Set12', \
                '/p/scratch/delia-mp/lin4/RestormerGrayScaleData/clean/test/BSD68', \
                '/p/scratch/delia-mp/lin4/RestormerGrayScaleData/clean/test/Urban100', \
    --distributed \
#    > ${LOG_DIR}/${EXPERIMENT}/record.txt 2>&1;
    
echo "Finish ADL experiment"
