import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import numpy as np
import cv2 
from joblib import Parallel, delayed
import multiprocessing

def load_gray_img(filepath):
    img = np.expand_dims(
        cv2.imread(
        filepath, cv2.IMREAD_GRAYSCALE
        ), axis=2
        )
    return img

root_in_dir = "/mnt/big_disk/s.lin/RestormerRGBData"
root_out_dir = "/mnt/big_disk/s.lin/RestormerGrayScaleData"

folds = [
    'train', 
    'test',
]

# Get all dirs
orginal_imgs_dirs = []
# for fold in folds:
#     if fold == "test":
#         root = ""
#         datasets = [
#             "BSD68", 
#             "Set12", 
#             "Urban100"
#             ]
#     elif fold == "train":
#         root = "DFWB"
#         datasets = [
#             "BSD400", 
#             "DIV2K", 
#             "Flickr2K", 
#             "WaterlooED"
#             ]
#     for dataset in datasets:
#         orginal_imgs_dirs.append(
#             os.path.join(root_in_dir, fold, root, dataset)
#         ) 

orginal_imgs_dirs.append(os.path.join(root_in_dir, "train_patches/DFWB"))

print("="*20)
print(orginal_imgs_dirs)
print("="*20)

def add_noise(img_file_rel, img_file_abs, out_dir, noise, sigma):
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
    cv2.imwrite(
        out_path,
        noisy_img_un_normalized
        )          


noises = [
    # "gaussian",
    # "rician",
    "mixed"
]

sigmas = [15,25,50]
sigmas = np.int_(sigmas)

num_cores = 20
for original_imgs_dir in orginal_imgs_dirs:
    print(f"input from {original_imgs_dir}")
    img_files_rel = [img for img in os.listdir(original_imgs_dir)] # GraySacleData/test/{dataset_name}
    img_files_abs = [
        os.path.abspath(os.path.join(original_imgs_dir, img_file_rel)) for img_file_rel in img_files_rel
        ]

    for noise in noises:
        for sigma in sigmas:
            sub_dir = f"{noise}_noise_{sigma}"
            
            inter_dir = os.path.relpath(original_imgs_dir, root_in_dir) # /test/{dataset_name}

            out_dir = os.path.join(root_out_dir, sub_dir, inter_dir) # GraySacleData/gaussian_noise_15/
            print(f"output to {out_dir}", end = "\t")
            os.makedirs(out_dir, exist_ok=True)

            Parallel(n_jobs=num_cores)(delayed(add_noise)(img_file_rel, img_file_abs, out_dir, noise, sigma) for img_file_rel, img_file_abs in tqdm(zip(img_files_rel, img_files_abs)))