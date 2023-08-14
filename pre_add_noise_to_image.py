import argparse
import yaml
import os
import sys
from collections import OrderedDict

import numpy as np
import cv2
import lpips

import torch

from utils import set_random_seed, OrderedYaml, load_gray_img, parse_irsde_cfg, dict_to_nonedict, preprocess
from model import load_model, inference_one_img
from utils.metrics import calculate_psnr, calculate_ssim
from utils.image_utils import np_img_to_torch, tensor2uint

from model.irsde.utils import tensor2img

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datasets", default='Set12,BSD68,Urban100', type=str)
    parser.add_argument("--sigmas", default='0,15,25,50', type=str, help='Sigma values')
    parser.add_argument("--in_dir", default="/mnt/big_disk/s.lin/RestormerRGB", type=str)
    parser.add_argument("--out_dir", default="/mnt/big_disk/s.lin/RestormerGrayScale", type=str)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    set_random_seed(args.seed)

    sigmas = np.int_(args.sigmas.split(','))

    print(sigmas)

    folds = [
        # 'train', 
        # 'val', 
        # 'test'
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
            
            for img_file_rel, img_file_abs in zip(img_files_rel, img_files_abs):
                # RGB -> gray
                img_gray_np = np.float64(load_gray_img(img_file_abs)) # (H,W,1)'

                # itereate through different level of noisy
                for sigma in sigmas:
                    if sigma == 0:
                        sub_dir = "clean"
                        out_dir = os.path.join(args.out_dir, sub_dir, fold, root, dataset)
                        os.makedirs(out_dir, exist_ok=True)

                        cv2.imwrite(
                            os.path.join(out_dir, img_file_rel),
                            img_gray_np
                            )                              
                    else:
                        sub_dir = f"noise_{sigma}"
                        out_dir = os.path.join(args.out_dir, sub_dir, fold, root, dataset)
                        os.makedirs(out_dir, exist_ok=True)

                        img_gray_np_ = img_gray_np/255.

                        np.random.seed(0)
                        noise = np.random.normal(0, sigma/255., img_gray_np.shape)
                        
                        noisy_img_gray_np = img_gray_np_ + noise 

                        noisy_img_gray_np = (noisy_img_gray_np.clip(0, 1) * 255.0).round()

                        # print(noisy_img_gray_np[1, :, :].flatten())

                        out_path = os.path.join(out_dir, img_file_rel)
                        # print(out_path)
                        cv2.imwrite(
                            out_path,
                            noisy_img_gray_np
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
        img_gray_np = np.float64(load_gray_img(img_file_abs)) # (H,W,1)

        # itereate through different level of noisy
        for sigma in sigmas:
            if sigma == 0:
                sub_dir = "clean"
                out_dir = os.path.join(args.out_dir, sub_dir, "train_patches", "DFWB")
                os.makedirs(out_dir, exist_ok=True)

                cv2.imwrite(
                    os.path.join(out_dir, img_file_rel),
                    img_gray_np
                    )                              
            else:
                sub_dir = f"noise_{sigma}"
                out_dir = os.path.join(args.out_dir, sub_dir, "train_patches", "DFWB")
                os.makedirs(out_dir, exist_ok=True)

                img_gray_np_ = img_gray_np/255.

                np.random.seed(0)
                noisy_img_gray_np = img_gray_np_ + np.random.normal(
                    0, 
                    sigma/255., 
                    img_gray_np_.shape
                    )
                
                noisy_img_gray_np = (noisy_img_gray_np.clip(0, 1) * 255.0).round()

                cv2.imwrite(
                    os.path.join(out_dir, img_file_rel),
                    noisy_img_gray_np
                    )           

if __name__ == '__main__':
    main()