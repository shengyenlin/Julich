export EXPERIMENT="0721_formal_fifty_epochs"
export NUM_CHANNELS=1

python3 inference.py \
    --model_dir /mnt/big_disk/s.lin/Experiment_archives/ADL/ \
    --test-dirs '/mnt/big_disk/s.lin/restormer_dataset/test/BSD68' \
    --EXPERIMENT ${EXPERIMENT} \
    --num-workers 1 \
    --json-file configs/ADL_test.json \
    --CHANNELS-NUM ${NUM_CHANNELS}