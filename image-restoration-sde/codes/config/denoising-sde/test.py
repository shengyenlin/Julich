import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
parser.add_argument("--weight_path", type=str)
parser.add_argument('--results', type=str, default='results', help='path of results')
parser.add_argument('--sigma', type=int, help='noise level: 15, 25, 50')
parser.add_argument('--input_dir', type=str)
parser.add_argument('--result_dir', type=str)

opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

args = parser.parse_args()

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
#### Input customized settings
# name, dataroot_GT, dataroot_LQ
datasets = [
    # 'Set12', 'BSD68', 
    'Urban100'
    ]

for dataset in datasets:
    opt["datasets"][dataset]['dataroot_LQ'] = os.path.join(
        args.input_dir,
        dataset
    )

    # just set for place holder, won't be used
    opt["datasets"][dataset]['dataroot_GT'] = opt["datasets"][dataset]['dataroot_LQ']

logger.info(option.dict2str(opt))  

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
opt["path"]["pretrain_model_G"] = args.weight_path
model = create_model(opt)
device = model.device

sde = util.DenoisingSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], device=device)
sde.set_model(model.model)

# degrad_sigma = opt["degradation"]["sigma"]

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(
        args.result_dir, test_set_name
        )
    util.mkdir(dataset_dir)

    test_times = []

    for ii, test_data in enumerate(test_loader):

        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        # GT = test_data["GT"]
        # LQ = util.add_noise(GT, degrad_sigma)

        # noisy_state = sde.noise_state(LQ) ??
        LQ = test_data["LQ"]
        # GT = test_data["GT"]

        model.feed_data(
            LQ, 
            # GT
            )
        tic = time.time()
        model.test(sde, sigma=args.sigma, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals(need_GT=False)
        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
        LQ = util.tensor2img(visuals["Input"].squeeze())  # uint8

        # GT = util.tensor2img(visuals["GT"].squeeze())  # uint8
        # suffix = opt["suffix"]
        # if suffix:
        #     save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        # else:
        #     save_img_path = os.path.join(dataset_dir, img_name + ".tif")
        
        save_img_path = os.path.join(dataset_dir, img_name + ".png")
        print(save_img_path)
        util.save_img(output, save_img_path)

        # LQ_img_path = os.path.join(
        #     dataset_dir, 
        #     img_name + \
        #      "_noisy.png"
        #     ".png"
        #         )
        # GT_img_path = os.path.join(dataset_dir, img_name + "_clean.png")
        # util.save_img(LQ, LQ_img_path)
        # util.save_img(GT, GT_img_path)


    print(f"average test time: {np.mean(test_times):.4f}")
