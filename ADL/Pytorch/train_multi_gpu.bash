echo "Start ADL experiment"

export EXPERIMENT="0804-rerun-gaussian-50-store-image"
export LOG_DIR="/mnt/big_disk/s.lin/Experiment_archives/ADL"
export CHANNELS_NUM=1 # gray->1, color->3

mkdir -p ${LOG_DIR}/${EXPERIMENT}/
touch ${LOG_DIR}/${EXPERIMENT}/record.txt 

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=1237

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=5,7 # need to check nvidia-smi first

python3 train.py \
	--EXPERIMENT ${EXPERIMENT} \
    --json-file configs/ADL_train.json \
    --DENOISER efficient_Unet \
    --effective_batch_size 16 \
    --world_size 2 \
    --noise_type gaussian \
    --noise_level 50 \
    --use_special_loss true \
	--CHANNELS-NUM ${CHANNELS_NUM} \
    --train-dirs '/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/train/DFWB' \
                '/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test/BSD68' \
    --test-dirs '/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test/Set12', \
                '/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test/BSD68', \
                '/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test/Urban100', \
    --distributed \
    > ${LOG_DIR}/${EXPERIMENT}/record.txt 2>&1;
    
echo "Finish ADL experiment"

### json file
# {
#     "model": "ADL",
#     "data":{
#         "H": 128,
#         "W": 128,
#         "batch_size": 0,
#         "task_mode": "DEN",
#         "noise_level": 50,
#         "shuffle": false,
#         "random_seed": 0,
#         "pin_memory": true,
#         "drop_last": true,
#         "num_valid_max": 256,
#         "localhost":null,
#         "img_types": ["png", "jpg", "jpeg", "bmp"],
#         "train_valid_ratio": 1, 
#         "use_customized_val_set": true,
#         "customized_val_set_len": 68,
#         "val_img_path_GT": "/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test/BSD68",
#         "val_img_path_LQ": "/mnt/big_disk/s.lin/RestormerGrayScaleData/mixed_noise_50/test/BSD68"
#     },
#     "ADL": {
#         "epochs": 50, 
#         "print_model": true,
#         "lr": 5e-5,
#         "optimizer":"Adam",
#         "lr_scheduler": {
#             "type": "MultiStepLR",
#             "kwargs": {
#                 "gamma": 0.8
#             }
#         }
#     },

#     "denoiser": {
#         "model": "Efficient_Unet",
#         "epochs": 50,
#         "print_model": true,
#         "lr": 1e-4,
#         "optimizer":"Adam",
#         "lr_scheduler": {
#             "type": "MultiStepLR",
#             "kwargs": {
#                 "gamma": 0.8
#             }
#         }
#     }


#     "discriminator": {
#         "model": "Efficient_Unet_disc",
#         "epochs": 50,
#         "print_model": true,
#         "lr": 1e-4,
#         "optimizer":"Adam",
#         "negative_slope":0.1,
#         "lr_scheduler": {
#             "type": "MultiStepLR",
#             "kwargs": {
#                 "gamma": 0.8
#             }
#         }
#     }
# }