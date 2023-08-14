export NCCL_DEBUG=INFO

python3 -m \
    torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=8888 \
    train.py \
    -opt=options/train/ir-sde-universal.yml \
    --launcher pytorch

echo 'Experiment End !'