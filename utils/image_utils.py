import numpy as np
import cv2
import skimage

import torch

def load_gray_img(filepath):
    img = np.expand_dims(
        cv2.imread(
        filepath, cv2.IMREAD_GRAYSCALE
        ), axis=2
        )
    return img

# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def np_img_to_torch(img):
    img = img/255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tr_img = torch.from_numpy(img).float().permute(2,0,1)
    return tr_img

def preprocess(model_name, file, sigma):
    
    img = np.float64(load_gray_img(file))

    if model_name == 'adl':
        WH = img.shape[0:2]

        # because of pyramid structure, the data size must be dividable by `blocks`
        blocks = 8
        WH = list(map(lambda x: (x//blocks)*blocks, WH))

        img = skimage.transform.resize(img, WH)
        # img = np.expand_dims(img, axis=-1) 

    gt_img = torch.from_numpy(
        img/255.
        ).permute(2,0,1)
    
    # add noise - torch
    noisy_img = gt_img + torch.randn_like(gt_img) * (sigma/255)
    noisy_img = noisy_img.to(torch.float32)

    return noisy_img, gt_img