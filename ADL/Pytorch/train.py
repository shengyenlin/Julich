from typing import Any, Callable, Dict, Tuple, Union
from argparse import ArgumentParser
import os
import time
import datetime
import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.distributed as dist


from model.denoiser_trainer import Denoiser_Trainer
from model.discriminator_trainer import Discriminator_Trainer
from model.adl_trainer import ADL_Trainer
from utils.dataloader import DataLoader_cls
from model.metric import MetricEval
from utils import util
from model import loss

import subprocess

# dir_path = os.path.dirname(os.path.abspath(__file__))

dir_path = os.environ['LOG_DIR']

#print(f"Torch Version: {torch.__version__}")
# available device
#num_gpus = torch.cuda.device_count()
#print(f"device_count: {num_gpus}")

###################################
parser = ArgumentParser()
parser.add_argument('--distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs.')
parser.add_argument('--num-workers', type=int, default=4, 
					help="number of workers.")
parser.add_argument('--DENOISER', type=str, default='.', action="store", required=True, help="...")
parser.add_argument('--train-dirs', nargs="+", default='trainDir', required=True, 
						help="Imoport your datasets. The datasets can be separated by comma")
parser.add_argument('--test-dirs', nargs="+", default='TestDir', required=True, help="...")
parser.add_argument('--EXPERIMENT', type=str, default='', action="store", required=True, help="...")
parser.add_argument('--json-file', type=str, action="store", default='', required=True, help="...")
parser.add_argument('--CHANNELS-NUM', type=int, action="store", default=0, required=True, help="...")

### My args
parser.add_argument("--effective_batch_size", type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--world_size", type=int)
parser.add_argument("--noise_type", type=str, choices=["gaussian", "rician", "mixed"])
parser.add_argument("--noise_level", type=int)
parser.add_argument("--use_special_loss", action="store_true")

args=parser.parse_args()

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_timestamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

class Train(object):
	def __init__(self,
				args:ArgumentParser, 
				config:Dict,
				loss_denoiser:Dict,
				eval_params_denoiser:Dict,
				loss_weights_denoiser:Dict,
				gpus:int):
		self.args = args
		self.config_data = config['data']
		self.gpus = gpus

		# Configure DistributedDataParallel (DDP) 
		ddp = util.struct_cls()
		ddp.rank = 0
		ddp.world_size = args.world_size
		self.ddp = ddp

		# Denoiser
		denoiser = util.struct_cls()
		denoiser.loss = loss_denoiser
		denoiser.loss_weights = loss_weights_denoiser
		denoiser.eval_params = eval_params_denoiser
		denoiser.config = config['denoiser']
		denoiser.config.update(config['STEPS'])
		denoiser.model = args.DENOISER
		self.denoiser = denoiser


		# Discriminator
		disc = util.struct_cls()
		disc.config = config['discriminator']
		disc.config.update(config['STEPS'])
		disc.model = disc.config['model']
		self.disc = disc

		# ADL
		adl = util.struct_cls()
		adl.model = config['model']
		adl.config = config['ADL']
		adl.config.update(config['STEPS'])
		self.adl = adl


	def __call__(self, PHASE):
		if PHASE == 1: 
			# get denoiser`s params
			params = util.prep(self.denoiser, 'denoiser', dir_path, args.EXPERIMENT)

		elif PHASE== 2:
			# get discriminator`s params
			params = util.prep(self.disc, 'discriminator', dir_path, args.EXPERIMENT)

		elif PHASE == 3:
			# get ADL`s params
			params = util.prep(self.adl, 'ADL', dir_path, args.EXPERIMENT)
			params.denoiser = util.prep(self.denoiser, 'denoiser', dir_path, args.EXPERIMENT)
			params.disc = util.prep(self.disc, 'discriminator', dir_path, args.EXPERIMENT)

		if self.args.distributed:
			#world_size = self.ddp.world_size * self.gpus
			print("Global settings")
			print(f"world_size: {args.world_size}, rank: {self.ddp.rank}")
			print("Distributed training...")

			mp.spawn(
				train_dist,args=(
							args.world_size, 
							self.args, 
							self.config_data, 
							self.ddp, 	
							params, PHASE
							),
						nprocs=args.world_size,
						join=True
				)
		else: 
			train(self.args, self.config_data, params, PHASE)

def train(args, config_data, params, PHASE):
	""" single gpu trainer"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using {} for training".format(device)) 

	# Load Data =============
	dataLoader_obj = DataLoader_cls(				batch_size=int(args.effective_batch_size),
				num_workers=int(args.num_workers),
				channels_num=args.CHANNELS_NUM,
				noise_type=args.noise_type,
				noise_level=args.noise_level,
				train_ds_dir=args.train_dirs,
				test_ds_dir=args.test_dirs,
				config=config_data,
				distributed=False
		)
	ds_train_loader, ds_valid_loader, ds_test_loader = dataLoader_obj()
	print('Train size: {} batches\nVal size: {} batches \nTest size: {} batches'.format(len(ds_train_loader), 
		len(ds_valid_loader), len(ds_test_loader)))

	if args.debug:
		for mode, ds in zip(['train', 'val', 'test'], [ds_train_loader, ds_valid_loader, ds_test_loader]):
			for batch_idx, data in enumerate(ds, 0):
				print('{}: {}-gpu: {}>>> {}, {}, [{:0.3f},{:0.3f}]'.format(mode, batch_idx, device,
							data['filename'], data['x'].size(),torch.min(data['y']),torch.max(data['y'])))
		exit()
		return


	# create writer =============
	writer_numerical = SummaryWriter(params.writer_numerical_dir)
	writer_imgs = SummaryWriter(params.writer_imgs_dir)

	if PHASE == 1:
		# Warmup denoiser =============
		denoiser_module= Denoiser_Trainer(params, device,
								args.CHANNELS_NUM, args.CHANNELS_NUM, 
								writer_numerical, writer_imgs, distributed=False)
		denoiser_module(ds_train_loader, ds_valid_loader, ds_test_loader)
	elif PHASE == 2:
		# Warmup discriminator =============
		discriminator_module= Discriminator_Trainer(params, device,
								args.CHANNELS_NUM, args.CHANNELS_NUM, 
								writer_numerical, writer_imgs, distributed=False)
		discriminator_module(ds_train_loader)

	elif PHASE == 3:
		# Run ADL =============
		adl_module= ADL_Trainer(params, device,
								args.CHANNELS_NUM, args.CHANNELS_NUM, 
								writer_numerical, writer_imgs, distributed=False)
		adl_module(ds_train_loader, ds_valid_loader, ds_test_loader)

def train_dist(gpu, ngpus_per_node, 
		args, config_data, ddp,
		params, PHASE):

	# Configure DDP =============
	ddp.gpu = gpu 
	ddp.rank = ddp.rank * ngpus_per_node + gpu
	# print(f"set device at: {ddp.gpu+5}")
	torch.cuda.set_device(ddp.gpu)

	print("Start a multiple-gpu trainer")
	print("Local settings")
	print(f"gpu id: {gpu}, world_size: {ddp.world_size}, rank: {ddp.rank}")

	dist.init_process_group(
		init_method="env://",
		backend='nccl', 
		timeout=datetime.timedelta(minutes=5), # prevent time out
		world_size=ddp.world_size, 
		rank=ddp.rank
		)

	if torch.distributed.is_initialized:
		print("Initalize process group correctly")

	batch_size = int(args.effective_batch_size / ngpus_per_node)
	num_workers = int(args.num_workers / ngpus_per_node)

	# Load Data =============
	dataLoader_obj = DataLoader_cls(batch_size=batch_size,
									num_workers=num_workers,
									channels_num=args.CHANNELS_NUM,
									noise_type=args.noise_type,
									noise_level=args.noise_level,
									train_ds_dir=args.train_dirs,
									test_ds_dir=args.test_dirs,
									config=config_data,
									distributed=True)
	ds_train_loader, ds_valid_loader, ds_test_loader = dataLoader_obj()
	if ddp.rank == 0:
		print('Train size: {} batches\nVal size: {} batches\nTest size: {} batches'.format(len(ds_train_loader), 
					len(ds_valid_loader), len(ds_test_loader)))

	if args.debug:
		
		for mode, ds in zip(['train', 'val', 'test'], [ds_train_loader, ds_valid_loader, ds_test_loader]):
			for batch_idx, data in enumerate(ds, 0):
				# print(data)
				print('{}: {}-gpu: {}>>> {} , {}'.format(
					mode, batch_idx, gpu,
					data['filename_y'], 
					data['y'].size(),))
				print("="*50)
		util.cleanup()
		exit(0)

	ds_train_loader.dataset.self_sanity_check()
	print("Train loader passed sanity check")
	ds_valid_loader.dataset.self_sanity_check()
	print("Valid loader passed sanity check")
	ds_test_loader.dataset.self_sanity_check()
	print("Test loader passed sanity check")
	print("*"*50)
	print("All dataloader pass sanity check!")

	# synchronize gpus =============
	dist.barrier()

	# Create writer =============
	if ddp.rank == 0:
		writer_numerical = SummaryWriter(params.writer_numerical_dir)
		writer_imgs = SummaryWriter(params.writer_imgs_dir)
	else:
		writer_numerical = writer_imgs = None


	if PHASE == 1:
		# Warmup denoiser =============
		denoiser_module= Denoiser_Trainer(
			params, ddp.rank,
			args.CHANNELS_NUM, args.CHANNELS_NUM, 
			writer_numerical, writer_imgs, distributed=True
			)
		
		denoiser_module(ds_train_loader, ds_valid_loader, ds_test_loader)

	elif PHASE == 2:
		# Warmup discriminator =============
		discriminator_module= Discriminator_Trainer(params, ddp.rank,
								args.CHANNELS_NUM, args.CHANNELS_NUM, 
								writer_numerical, writer_imgs, distributed=True)
		discriminator_module(ds_train_loader)

	elif PHASE == 3:
		# Run ADL =============
		adl_module= ADL_Trainer(params, ddp.rank,
								args.CHANNELS_NUM, args.CHANNELS_NUM, 
								writer_numerical, writer_imgs, distributed=True)
		adl_module(ds_train_loader, ds_valid_loader, ds_test_loader)
	
	util.cleanup()



if __name__== '__main__':
	### Original config ###
	# and args

	config = util.read_config(args.json_file)


	print('configuration', '*'*20)
	for key, item in config.items(): print(key, item) 
	print('*'*20)
	#######################

	set_random_seed(config['data']['random_seed'])

	ngpus_per_node = torch.cuda.device_count()
	for i in range(ngpus_per_node):
		gpu_device = torch.cuda.get_device_name(i)
		gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert bytes to GB
		print(f"GPU {i+1}: {gpu_device}, Memory: {gpu_memory:.2f}GB")

	if args.distributed:
		assert dist.is_available()

	# Evaluation metrics and fidelity term for generator/denoiser

	if args.use_special_loss:
		loss_denoiser_fns = {
		'L1': loss.Loss_L1, 
		'Histgram': loss.Hist_loss, 
		'atw-edge': loss.pyr_Loss
		}
		print("="*10, "use special loss", "="*10)
		loss_weights_denoiser = {
			'L1': 1., 
			'Histgram': 1.,  # histogram loss
			'atw-edge': 1. # pyramidal textural loss
			}
	else:
		print("="*10, "use only L1 loss", "="*10)
		loss_denoiser_fns = {
		'L1': loss.Loss_L1, 
		}
		loss_weights_denoiser = {
			'L1': 1., 
			}

	eval_params_denoiser = {
		'psnr': MetricEval.psnr,
	}


	print('Let`s start training the model', '.'*3)

	Trainer = Train(args=args, 
					config=config, 
					loss_denoiser=loss_denoiser_fns, 
					eval_params_denoiser = eval_params_denoiser,
					loss_weights_denoiser = loss_weights_denoiser,
					gpus=ngpus_per_node)

	ticks = time.time()
	Trainer(PHASE=1) # 1: warm-up denoiser
	Trainer(PHASE=2) # 2: warm-up discriminator
	Trainer(PHASE=3) # 3: ADL
	elapsed_time = time.time() - ticks
	print('[i] Training took hh:mm:ss->{} (hh:mm:ss).'.format(
				str(datetime.timedelta(seconds=elapsed_time)))) 

	print('Done!')
