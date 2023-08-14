# 2D image denoising model

## Introduction
This repo stores four different kinds of 2D image denoising model
- DnCNN (KAIR): CNN-based
- Restormer: transformer-based
- ADL: GAN-based
- Image-restoration-sde: diffusion-based

## Testing 
```
bash test.bash <model_name> <noise_type> <noise_level> <config_path> <weight_path>

bash test.bash irsde gaussian 50 \
    image-restoration-sde/codes/config/denoising-sde/options/test/ir-sde.yml \
    weights/irsde-denoise-gaussian-50.pth;

bash test.bash adl gaussian 50 \
    ADL/Pytorch/configs/ADL_test.json \
    weights/adl-gaussian-50.pt;

bash test.bash restormer gaussian 50 \
    KAIR/options/train_dncnn-b-universal.json \
    weights/dncnn-gaussian-50.pth;

bash test.bash dncnn gaussian 50 \
    KAIR/options/train_dncnn-b-universal.json \
    weights/dncnn-gaussian-50.pth;
```



### Restormer
1. Specify 

## Evaluation