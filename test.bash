#!/bin/bash
# Read noisy images and output clean images with specified models

export CUDA_VISIBLE_DEVICES=3

noise_root=/mnt/big_disk/s.lin/RestormerGrayScaleData
result_root=/mnt/big_disk/s.lin/ModelZooResult

model=$1 # irsde, adl, restormer, dncnn
noise_type=$2 # gaussian, rician, mixed
noise_level=$3 # 15, 25, 50
config_path=$4
weight_path=$5

noisy_img_dir=${noise_root}/${noise_type}_noise_${noise_level}/test
result_dir=${result_root}/${model}/${noise_type}_noise_${noise_level}

# echo ${noisy_img_dir}

if [ "$model" = "restormer" ]; then
    # Restormer
    # ok
    python3 ./Restormer/Denoising/test_gaussian_gray_denoising.py \
        --input_dir ${noisy_img_dir} \
        --config_path ${config_path} \
        --result_dir ${result_dir} \
        --weights_path ${weight_path} \
        --sigmas ${noise_level};

elif [ "$model" = "dncnn" ]; then
    # DnCNN
    # ok
    python3 ./KAIR/main_test_dncnn.py \
        --need_degradation false \
        --model_name dncnn3 \
        --results ${result_dir} \
        --sigma ${noise_level} \
        --testsets ${noisy_img_dir} \
        --weight_path ${weight_path};


elif [ "$model" = "irsde" ]; then
    # IR-SDE
    # Add your IR-SDE command here
    python image-restoration-sde/codes/config/denoising-sde/test.py \
        -opt ${config_path} \
        --input_dir ${noisy_img_dir} \
        --result_dir ${result_dir} \
        --weight_path ${weight_path} \
        --sigma ${noise_level};

elif [ "$model" = "adl" ]; then
    # ADL
    python3 ADL/Pytorch/inference.py \
        --result_dir ${result_dir} \
        --test-dirs ${noisy_img_dir} \
        --json-file ${config_path} \
        --weight_path ${weight_path} \
        --CHANNELS-NUM 1;

else
    echo "Invalid model option. Please choose 'restormer', 'dncnn', 'irsde', or 'adl'."
fi