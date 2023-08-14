from typing import Any, Callable, Dict, Tuple, Union, Iterable
import random
import numpy as np
import cv2

import torch
from torchvision import transforms




class ToTensor_fn(object):
    def __call__(self, sample):
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] = torch.from_numpy(sample[k])
        return sample


class Normalization(object):
    def __call__(self, sample):
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] /= 255.
        return sample
        

class ImRotate90(object):
    def __init__(self, p):
        self.p = p
    
    def __call__(self, sample):
        assert 1>=self.p>=0, 'p is limited in [0 1]'
        p = self.p*100
        
        if np.random.randint(100, size=1)<= p:
            rand_ =  np.random.randint(1,4, size=1)
            for k in sample.keys():
                if (sample[k] is not False) and (not isinstance(sample[k], str)):
                    sample[k] = np.rot90(sample[k], k=rand_, axes=(0, 1)).copy() 
        return sample




class AddGaussianNoise(object):
    def __init__(self, noise_level:Union[Iterable[float],float], Training:bool):
        for noise_ in noise_level:
            assert noise_ >= 0., 'Enter valid noise level!'

        if Training:
            self.noise_level = np.random.uniform(low=noise_level[0], high=noise_level[1], size=(1,))   
        else:
            self.noise_level = np.max(noise_level)# get the maximum noise value for test

    def __call__(self, sample):
        for key in sample.keys():
            if sample['x'] is not False:
                sample['y'] = sample['x'] + np.random.normal(loc=0.0, 
                                                            scale=self.noise_level/255., 
                                                            size=(sample['x'].shape)
                                                            )
        return sample



class ImFlip_lr(object):
    def __init__(self,p):
        self.p = p
        
    def __call__(self, sample):
        
        assert 1>=self.p>=0, 'p is limited in [0 1]'
        p = self.p*100
        if np.random.randint(100, size=1).item()<= p:
            for k in sample.keys():
                if (sample[k] is not False) and (not isinstance(sample[k], str)):
                    sample[k] = np.fliplr(sample[k]).copy() 
        return sample
          


class ImFlip_ud(object):
    def __init__(self,p):
        self.p = p
        
    def __call__(self, sample):
        assert 1>=self.p>=0, 'p is limited in [0 1]'
        p = self.p*100
        
        if np.random.randint(100, size=1).item()<= p:
            for k in sample.keys():
                if (sample[k] is not False) and (not isinstance(sample[k], str)):
                    sample[k] = np.flipud(sample[k]).copy() 
        return sample


    
class Channel_transpose(object):
    def __init__(self, transpose_tuple):
        assert isinstance(transpose_tuple, tuple) and len(transpose_tuple)==3, "Invalid transposed tuple" 
        self.transpose_tuple = transpose_tuple

    def __call__(self, sample):
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] = np.transpose(sample[k], self.transpose_tuple) 
        return sample
  

# Transforms =======================================
def Transform_training(noise_level:Union[Iterable[float], float], Training:bool, transpose_tuple=(2,0,1), p=0.5):
    return transforms.Compose([Normalization(), 
                                ImFlip_lr(p), 
                                ImFlip_ud(p), 
                                AddGaussianNoise(noise_level, Training),
                                Channel_transpose(transpose_tuple), 
                                ToTensor_fn(),
        ]) 


def Test_denoising(noise_level:Union[list,float], Training:bool, transpose_tuple=(2,0,1), p=0.5): 
    return transforms.Compose([Channel_transpose(transpose_tuple), 
                                ToTensor_fn()]
    )

# from restormer

def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)