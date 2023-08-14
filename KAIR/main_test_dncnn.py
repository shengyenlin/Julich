import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo          # model_zoo
   |--dncnn_15        # model_name
   |--dncnn_25
   |--dncnn_50
   |--dncnn_gray_blind
   |--dncnn_color_blind
   |--dncnn3
|--testset            # testsets
   |--set12           # testset_name
   |--bsd68
   |--cbsd68
|--results            # results
   |--set12_dncnn_15  # result_name = testset_name + '_' + model_name
   |--set12_dncnn_25
   |--bsd68_dncnn_15
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dncnn_25', help='dncnn_15, dncnn_25, dncnn_50, dncnn_gray_blind, dncnn_color_blind, dncnn3')
    parser.add_argument('--sigma', type=int, help='noise level: 15, 25, 50')
    parser.add_argument('--model_pool', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--need_degradation', type=bool, default=True, help='add noise or not')
    parser.add_argument('--task_current', type=str, default='dn', help='dn for denoising, fixed!')
    
    args = parser.parse_args()

    if 'color' in args.model_name:
        n_channels = 3        # fixed, 1 for grayscale image, 3 for color image
    else:
        n_channels = 1        # fixed for grayscale image
    if args.model_name in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
        nb = 20               # fixed
    else:
        nb = 17               # fixed

    # model_path = os.path.join(args.model_pool, args.model_name+'.pth')
    model_path = args.weight_path

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    datasets = ['Set12', 'BSD68', 'Urban100']
    for dataset in datasets:
        L_path = os.path.join(args.testsets, dataset) # L_path, for Low-quality images
        H_path = L_path                               # H_path, for High-quality images
        E_path = os.path.join(args.results, dataset)   # E_path, for Estimated images
        util.mkdir(E_path)
        
        logger_name = str(dataset)
        utils_logger.logger_info(logger_name)
        logger = logging.getLogger(logger_name)

        need_H = True if H_path is not None else False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ----------------------------------------
        # load model
        # ----------------------------------------

        from models.network_dncnn import DnCNN as net
        # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
        model.load_state_dict(torch.load(model_path), strict=True)
        

        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        
        logger.info('Model path: {:s}'.format(model_path))
        number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('Params number: {}'.format(number_parameters))

        logger.info('model_name:{}, image sigma:{}'.format(args.model_name, args.sigma))
        logger.info(L_path)
        L_paths = util.get_image_paths(L_path)
        # H_paths = util.get_image_paths(H_path) if need_H else None

        for idx, img in enumerate(L_paths):

            # ------------------------------------
            # (1) img_L
            # ------------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))

            img_L = util.imread_uint(img, n_channels=n_channels)
            img_L = util.uint2single(img_L)

            #  HWC np -> bs, CHW torch
            img_L = util.single2tensor4(img_L)
            img_L = img_L.to(device)

            img_E = model(img_L)
            img_E = util.tensor2uint(img_E)

            # ------------------------------------
            # save results
            # ------------------------------------

            util.imsave(img_E, os.path.join(E_path, img_name+ext))

if __name__ == '__main__':

    main()
