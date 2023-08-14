import argparse
import yaml
import os
import sys
from collections import OrderedDict

import numpy as np
import cv2
import lpips

import torch

from image_utils import load_gray_img

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datasets", default='Set12,BSD68,Urban100', type=str)
    parser.add_argument("--sigmas", default='15,25,50', type=str, help='Sigma values')
    parser.add_argument("--in_dir", default="/mnt/big_disk/s.lin/RestormerRGBData", type=str)
    parser.add_argument("--out_dir", default="/mnt/big_disk/s.lin/RestormerGrayScaleData", type=str)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    sigmas = np.int_(args.sigmas.split(','))

    folds = [
        # 'train', 
        # 'val', 
        'test'
        ]

    noises = [
        # "gaussian"
        "rician",
        "mixed"
    ]

    # itereate through different folds of dataset
    for fold in folds:
        if fold == "test":
            root = ""
            datasets = [
                "BSD68", 
                "Set12", 
                "Urban100"
                ]
        elif fold == "train":
            root = "DFWB"
            datasets = ["BSD400", "DIV2K", "Flickr2K", "WaterlooED"]

        # read img
        for dataset in datasets:
            in_dir = os.path.join(
                args.in_dir, fold, root, dataset
            )

            print(f"Currently dealing with {in_dir}")

            img_files_rel = [img for img in os.listdir(in_dir)]
            img_files_abs = [
                os.path.abspath(os.path.join(in_dir, img_file_rel)) for img_file_rel in img_files_rel
                ]

            for noise in noises:
                # itereate through different level of noisy
                for sigma in sigmas:
                    print(f"Noise: {noise}, sigma: {sigma}")
                    sub_dir = f"{noise}_noise_{sigma}"
                    out_dir = os.path.join(args.out_dir, sub_dir, fold, root, dataset)
                    os.makedirs(out_dir, exist_ok=True)

                    for img_file_rel, img_file_abs in zip(img_files_rel, img_files_abs):
                        # RGB -> gray
                        original_img = np.float64(load_gray_img(img_file_abs)) # (H,W,1)
                        original_img_ = original_img/255.

                        # make sure noise added to all images are the same
                        np.random.seed(0) 
                        H, W, C = original_img.shape[0], original_img.shape[1], original_img.shape[2]
                        N = H * W * C

                        # rician noise
                        v = 0
                        rician_noise = np.random.normal(
                            scale=sigma/255, size=(N, 2) # 2 = real and imaginary parts of complex-valued Rician
                            ) + [[v/255,0]] # add v to all els in the real part, create a non-central Rician distribution
                        rician_noise = np.linalg.norm(rician_noise, axis=1) # compute L2-norm along dim = 1, shape = (N,)
                        rician_noise = rician_noise.reshape(original_img_.shape)

                        # gaussian noise
                        gaussian_noise = np.random.normal(0, sigma/255., original_img.shape)

                        if noise == "gaussian":
                            noisy_img = original_img_ + gaussian_noise
                        
                        elif noise == "rician":
                            noisy_img = original_img_ + rician_noise

                        elif noise == "mixed":
                            noisy_img = original_img_ + rician_noise + gaussian_noise

                        noisy_img_un_normalized = (noisy_img.clip(0, 1) * 255.0).round()

                        out_path = os.path.join(out_dir, img_file_rel)
                        # print(out_path)

                        cv2.imwrite(
                            out_path,
                            noisy_img_un_normalized
                            )          

    # dealing with train_patches
    patches_dir = os.path.join(args.in_dir, "train_patches", "DFWB")
    print(f"Currently dealing with {patches_dir}")
    img_files_rel = [img for img in os.listdir(patches_dir)]
    img_files_abs = [
        os.path.abspath(os.path.join(patches_dir, img_file_rel)) for img_file_rel in img_files_rel
        ]

    for img_file_rel, img_file_abs in zip(img_files_rel, img_files_abs):
        # RGB -> gray
        original_img = np.float64(load_gray_img(img_file_abs)) # (H,W,1)

        for noise in noises:
            # itereate through different level of noisy
            for sigma in sigmas:                       
                print(f"Noise: {noise}, sigma: {sigma}")   
                sub_dir = f"{noise}_noise_{sigma}"
                out_dir = os.path.join(args.out_dir, sub_dir, "train_patches", "DFWB")
                os.makedirs(out_dir, exist_ok=True)

                original_img_ = original_img/255.

                # make sure noise added to all images are the same
                np.random.seed(0) 
                H, W, C = original_img.shape[0], original_img.shape[1], original_img.shape[2]
                N = H * W * C

                # rician noise
                v = 0
                rician_noise = np.random.normal(
                    scale=sigma/255, size=(N, 2) # 2 = real and imaginary parts of complex-valued Rician
                    ) + [[v/255,0]] # add v to all els in the real part, create a non-central Rician distribution
                rician_noise = np.linalg.norm(rician_noise, axis=1) # compute L2-norm along dim = 1, shape = (N,)
                rician_noise = rician_noise.reshape(original_img_.shape)

                # gaussian noise
                gaussian_noise = np.random.normal(0, sigma/255., original_img.shape)

                if noise == "gaussian":
                    noisy_img = original_img_ + gaussian_noise
                
                elif noise == "rician":
                    noisy_img = original_img_ + rician_noise

                elif noise == "mixed":
                    noisy_img = original_img_ + rician_noise + gaussian_noise

                noisy_img_un_normalized = (noisy_img.clip(0, 1) * 255.0).round()

                cv2.imwrite(
                    os.path.join(out_dir, img_file_rel),
                    noisy_img_un_normalized
                )           

if __name__ == '__main__':
    main()