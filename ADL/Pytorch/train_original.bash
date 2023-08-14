#### !!!!!: Need to change to world size
#### !!!!!: Need to check nvidia-smi first

echo "Start ADL experiment"

export EXPERIMENT="0724_original_setting"
export CHANNELS_NUM=1 # gray->1, color->3
export LOG_DIR="/mnt/big_disk/s.lin/Experiment_archives/ADL"

# export MASTER_ADDR=134.94.155.141
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=5678

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=1,2
# export CUDA_LAUNCH_BLOCKING=1

mkdir -p ${LOG_DIR}/${EXPERIMENT}
touch ${LOG_DIR}/${EXPERIMENT}/logs.txt

python3 train.py \
    --DENOISER efficient_Unet \
    --effective_batch_size 4 \
    --world_size 2 \
	--EXPERIMENT ${EXPERIMENT} \
	--json-file configs/ADL_train_original.json \
	--CHANNELS-NUM ${CHANNELS_NUM} \
    --train-dirs '/mnt/big_disk/s.lin/restormer_dataset/train/DFWB/Flickr2K' \
    --test-dirs '/mnt/big_disk/s.lin/restormer_dataset/test/BSD68', \
    --distributed \
    > ${LOG_DIR}/${EXPERIMENT}/logs.txt 2>&1;


echo "Finish ADL experiment"
