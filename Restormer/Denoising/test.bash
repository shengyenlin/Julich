python3 test_gaussian_gray_denoising.py \
    --weights /home/s.lin/Restormer/Denoising/pretrained_models/restormer-0725-original-setting \
    --input_dir /mnt/big_disk/s.lin/restormer_dataset/test \
    --result_dir ./results/tmp/ \
    --model_type blind \
    --sigmas 15,25,50;