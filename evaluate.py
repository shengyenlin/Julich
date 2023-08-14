import argparse
from collections import OrderedDict
import os 

import skimage
import numpy as np
import cv2
import torch
import lpips
from torchvision.models import AlexNet_Weights


from utils.metrics import calculate_psnr, calculate_ssim
from utils.image_utils import load_gray_img, np_img_to_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str)
    parser.add_argument("--datasets", default='Set12,BSD68,Urban100', type=str)
    parser.add_argument("--sigmas", default='50', type=str, help='Sigma values')
    parser.add_argument("--noise_types", type=str)
    parser.add_argument("--gt_dir", type=str, default="/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test")
    parser.add_argument("--setting",type=str)
    parser.add_argument("--pred_dir", type=str, default="/mnt/big_disk/s.lin/ModelZooResult/")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"

    lpips_fn = lpips.LPIPS(
        # net='alex',
        # weight=AlexNet_Weights.IMAGENET1K_V1
        ).to(device)

    sigmas = np.int_(args.sigmas.split(','))
    datasets = args.datasets.split(',')
    noise_types = args.noise_types.split(',')
    models = args.models.split(',')

    print()
    print("Start evaluation!")

    for model in models:
        if model == "adl":
            args.gt_dir = "/mnt/big_disk/s.lin/RestormerGrayScaleData/clean_for_ADL/test"
        else:
            args.gt_dir = "/mnt/big_disk/s.lin/RestormerGrayScaleData/clean/test"

    
        print(f"====================== Model: {model} ======================")

        for noise_type in noise_types:
            print("-----------------------")
            print(f"|Noise type: {noise_type}|")
            print("-----------------------")
            print("\t\t\t PSNR \t   SSIM      LPIPS")
            for dataset in datasets:
                # Load gt image
                gt_dir = os.path.join(args.gt_dir, dataset)
                gt_files = [f for f in os.listdir(gt_dir)]
                gt_files = [os.path.join(gt_dir, file) for file in gt_files]
                # normalize imgs
                gt_imgs = [
                    load_gray_img(file) for file in gt_files
                    # cv2.imread(file) for file in gt_files
                ]

                for sigma in sigmas:    
                    
                    pred_img_dir = os.path.join(
                        args.pred_dir, f"{model}", 
                        # args.setting,
                        f"{noise_type}_noise_{sigma}", dataset)

                    test_results = OrderedDict()
                    test_results["psnr"] = []
                    test_results["ssim"] = []
                    test_results["psnr_y"] = []
                    test_results["ssim_y"] = []
                    test_results["lpips"] = []

                    pred_files = [f for f in os.listdir(pred_img_dir)]
                    pred_files = [os.path.join(pred_img_dir, file) for file in pred_files]

                    # print(pred_files)
                    pred_imgs = [
                        load_gray_img(file) for file in pred_files
                    ]

                    cnt = 0
                    for gt_img, pred_img in zip(gt_imgs, pred_imgs):
                        # print(gt_files[cnt], pred_files[cnt])
                        psnr = calculate_psnr(pred_img, gt_img)
                        ssim = calculate_ssim(pred_img, gt_img)

                        lp_score = lpips_fn(
                            np_img_to_torch(gt_img).to(device) * 2 - 1, # normalize, CHW->HWC
                            np_img_to_torch(pred_img).to(device) * 2 - 1
                        ).squeeze().item()

                        test_results["psnr"].append(psnr)
                        test_results["ssim"].append(ssim)
                        test_results["lpips"].append(lp_score)
                        cnt+=1

                    avg_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
                    avg_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
                    avg_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])

            
                    print(f"(level: {sigma}) {dataset}", end=' - ')
                    print("\t {:.3f}    {:.3f}     {:.3f}".format(avg_psnr, avg_ssim, avg_lpips))

if __name__ == '__main__':
    main()