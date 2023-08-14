from typing import Any, Callable, Dict, Tuple, Union, Iterable
import os
import numpy as np
import random
import re

import cv2
from skimage import io, color, transform

import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
import warnings
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler



from utils.transform_collections import  Transform_training, paired_random_crop, random_augmentation, img2tensor


class DataLoader_cls(Dataset): 
    def __init__(self,
                num_workers: int,
                batch_size:int,
                channels_num: int,
                noise_type: str,
                noise_level: int,
                train_ds_dir: Union[str,list],
                test_ds_dir: Union[str,list],
                config:  Union[str,list],
                distributed:bool=False,
        )->Union[DataLoader, DataLoader, DataLoader]: 
        r"""Data loader using DDP
        
        Args:
            num_workers: number of workers. It will be divided by the number of gpus
            batch_size: the size of batches. It will be divided by the number of gpus
            channels_num: input channels (RGB:3, grey:1)
            train_ds_dir: A list of directories for training datasets
            test_ds_dir: A list of directories for test datasets
            config: configuration file for data
        """

        self.distributed = distributed
        
        # chech the directories
        self.train_dir = _get_dir(train_ds_dir) 
        self.test_dir = _get_dir(test_ds_dir)


        # dataloader params
        self.config = config
        self.shuffle = config['shuffle']
        self.train_valid_ratio = config['train_valid_ratio']
        self.num_valid_max = config['num_valid_max']
        self.random_seed = config['random_seed']


        self.DL_params = {'batch_size': batch_size,
                        'num_workers': num_workers,
                        'pin_memory': config['pin_memory'],
                        'drop_last':  config['drop_last']
                    }
        
        self.noise_type = noise_type
        self.noise_level = noise_level

        # x: ground-truth, y: noisy sample 
        # (if False, we will add synthesized noise to x)
        self.data_mode = {
            'y':True, 
            'x':True, 
            'mask':False, 
            'filename': False
            }

        self.DS_params = {'data_mode': self.data_mode, 
                        'task_mode': config['task_mode'] ,
                        'WHC': [config['W'],config['H'],channels_num], 
                        'img_format': config['img_types'],
                    }

        # Customized validation dataset
        self.use_customized_val_set = self.config['use_customized_val_set']
        if self.use_customized_val_set:
            self.substring_GT = "/p/scratch/delia-mp/lin4/RestormerGrayScaleData/clean/test/BSD68"
            self.substring_LQ = self.substring_GT.replace("/clean/", f"/{noise_type}_noise_{noise_level}/")
            self.customized_val_set_len = self.config["customized_val_set_len"]
            print("GT Path")
            print(self.substring_GT)
            print("LQ Path")
            print(self.substring_LQ)

    def __call__(self):
        
        # get train & validation datasets
        # img_dirs['x'], img_dirs['y']
        img_files = _get_files(
            self.train_dir, 
            self.noise_level,
            self.noise_type,
            self.config['img_types'], 
            self.data_mode
            )

        # divide the datasets into train and validation
        train_idx, valid_idx = self._train_valid_sampler(
            len(img_files['x'])
            )

        # train-valid set check
        if self.use_customized_val_set:
            print(f"GT dir: {self.substring_GT}")
            print(f"LQ dir: {self.substring_LQ}")
            for sign in ['x', 'y']:
                img_filename = img_files[sign]

                train_files = [img_filename[idx] for idx in train_idx]
                valid_files = [img_filename[idx] for idx in valid_idx]

                if sign == 'x':
                    train_not_contain_substring = not any(self.substring_GT in element for element in train_files)
                    valid_all_contains_substring = all(self.substring_GT in element for element in valid_files)

                elif sign == 'y':
                    train_not_contain_substring = not any(self.substring_LQ in element for element in train_files)
                    valid_all_contains_substring = all(self.substring_LQ in element for element in valid_files)                    

                assert train_not_contain_substring and valid_all_contains_substring
            print("Pass train valid dataset check")
        
        if self.distributed:  
            # create Dataset_cls: TRAIN
            img_files_train = {
                key:[img_files[key][idx] for idx in train_idx] 
                            for key, val in self.data_mode.items() if val}
            
            Dataset_Train = Dataset_cls(
                img_files=img_files_train, 
                Training=True, 
                noise_type=self.noise_type,
                noise_level=self.noise_level, 
                **self.DS_params
                )
            
            # create Dataset_cls: VALID
            img_files_valid = {key:[img_files[key][idx] for idx in valid_idx] 
                                for key, val in self.data_mode.items() if val}
            Dataset_Valid = Dataset_cls(img_files=img_files_valid, 
                                    # Training=True, 
                                    Training=False, 
                                    noise_type=self.noise_type,
                                    noise_level=self.noise_level, 
                                    **self.DS_params)

            # create sampler
            train_sampler = DistributedSampler(
                Dataset_Train, 
                drop_last=self.config['drop_last'], 
                seed=self.random_seed, 
                shuffle=self.shuffle
                )
            
            valid_sampler = DistributedSampler(
                Dataset_Valid, 
                drop_last=self.config['drop_last'], 
                seed=self.random_seed, 
                shuffle=self.shuffle
                )

            # create data loader
            train_loader = DataLoader(dataset= Dataset_Train, sampler= train_sampler, 
                                    **self.DL_params )
            valid_loader = DataLoader(dataset= Dataset_Valid, sampler= valid_sampler, 
                                    collate_fn=collate_fn, **self.DL_params )

        else:
            # create Dataset_cls
            Dataset_Train = Dataset_cls(img_files=img_files, 
                                    Training=True, 
                                    noise_type=self.noise_type,
                                    noise_level=self.noise_level, 
                                    **self.DS_params)

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset= Dataset_Train, 
                sampler= train_sampler, 
                shuffle=False,
                **self.DL_params
                )
            valid_loader = DataLoader(
                dataset= Dataset_Train, 
                sampler= valid_sampler, 
                collate_fn=collate_fn, 
                **self.DL_params
                )


        # create dataloader for test
        img_files = _get_files(
            self.test_dir, 
            self.noise_level,
            self.noise_type,
            self.config['img_types'], 
            self.data_mode
            )
        Dataset_Test  = Dataset_cls(img_files=img_files,  
                                    Training=False, 
                                    noise_type=self.noise_type,
                                    noise_level=self.noise_level, # not using
                                    **self.DS_params)
        test_loader  = DataLoader(dataset= Dataset_Test, shuffle= False, 
                                    collate_fn=collate_fn, **self.DL_params)

        return  train_loader, valid_loader, test_loader



    def _train_valid_sampler(self, len_train_valid):
        """sampler for training and validation"""

        split = len_train_valid - int(self.train_valid_ratio * len_train_valid)

        # check the upper limit of validation samples
        if split > self.num_valid_max:
            split = self.num_valid_max

        indices = list(range(len_train_valid))

        if self.use_customized_val_set:
            train_indices = indices[:-self.customized_val_set_len]
            val_indices = indices[-self.customized_val_set_len:]

        if self.shuffle:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            np.random.seed(self.random_seed)
            if self.use_customized_val_set: 
                np.random.shuffle(train_indices)
            else:
                np.random.shuffle(indices)
        
        if not self.use_customized_val_set:
            train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices
    

def collate_fn(batch):
    """remove bad samples"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

 
def _get_dir(_dir:Callable[[Union[list,str]],str])->list:
    """check and get directories"""
    if type(_dir)== str:
        _dir = list([_dir,])
    
    dirs = []
    for item in _dir:
        dirs.append("".join(list(map(lambda c: c if c not in r'[,*?!:"<>|] \\' else '', item))))
    return dirs


def _initilize_data_mode(data_mode:Dict[str,bool])->Dict[str,Any]:
    """ Ininitlize data mode by the input data mode"""

    data_mode_ = {key: False for key in ['y', 'x', 'mask']} 
    if data_mode is not None:
        data_mode_.update(data_mode)
    
    return data_mode_

def _get_files(dirs_:Union[list,Iterable[str]], noise_level, noise_type, img_format, data_mode):
    """ get image files
        in-args:
            dirs_: list of data directories
        out-args: 
            img_dirs: list of the address of all avail images
    """
    dirs_ = list(dirs_)
    img_dirs = _initilize_data_mode(data_mode)

    # if data_mode['y']:
    #     dirs_ = [os.path.join(dir_,'HR') for dir_ in dirs_] 

    # GT - clean
    img_dirs['x'] = [os.path.join(path, name) 
                    for dir_i in dirs_ 
                        for path, subdirs, files_ in os.walk(dir_i)  
                            for name in files_ 
                                if name.lower().endswith(tuple(img_format))]

    # LQ - noisy
    if data_mode['y']:
        img_dirs['y'] =  [
            files.replace('/clean/', f'/{noise_type}_noise_{noise_level}/') for files in img_dirs['x']
            ]

    # if data_mode['mask']:
    #     img_dirs['mask'] =  [files.replace('/HR/', '/mask/') for files in x_files]

    return img_dirs



def _im_read_resize(PATH:str, WHC:Tuple[int, Iterable[int]]):
    """read and resize images"""

    assert isinstance(WHC, (list, tuple)) and len(WHC)==3, "Invalid tuple for width, height, channel"
    Width, Height, Channel = WHC[0], WHC[1], WHC[2] # image shape required (after being resized)

    # sync as restormer
    img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.

    WH = img.shape[0:2]

    # because of pyramid structure, the data size must be dividable by `blocks`
    blocks = 8
    WH = list(map(lambda x: (x//blocks)*blocks, WH))
    img = transform.resize(img, WH)

    # ./train/DFWB/WaterlooED/00027.bmp, shape = (500, 676, 4)
    # the last axis of the last dim. is all zero
    if img.shape[-1] == 4:  
        img = img[:, :, 0:3]

    # img.shape[-1] == 3, only for rgb images
    # testing set of BSD doesn't contain channel dimension
    if Channel == 1 and img.shape[-1] == 3:
        img = color.rgb2gray(img)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1) 

    if Width > 0 and Height > 0:
        img = transform.resize(img, (Height, Width)) 

    return img




class Dataset_cls(Dataset):
    def __init__(self,
                img_files: str,
                Training: bool,
                data_mode:Dict[str, bool],
                task_mode:str,
                noise_type: str,
                noise_level: Union[list,float],
                WHC:list,
                img_format:list,
                keep_last_n_dirs:int=6
                ):
        r""" Dataset class for one dataset

        Args:
            img_files: lsit of filenames
            Training: whether testing or training
            data_mode: type of input images
            task_mode: DEN=Denoising, etc
            noise_level: the level of noise
            WHC: [Width, height, depth]
            img_format: extension of images
            keep_last_n_dirs: save last n directories of a filename
        """
        super(Dataset, self).__init__()

        self.data_mode = _initilize_data_mode(data_mode)
        self.task_mode = task_mode
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.WHC = WHC
        self.Training = Training
        self.img_files = img_files 
        self.keep_last_n_dirs = keep_last_n_dirs
        
    def __len__(self):
        return len(self.img_files['x'])


    def sort_file_name(self):
        self.img_files['x'] = sorted(self.img_files['x'])
        self.img_files['y'] = sorted(self.img_files['y'])

    def print_img_files(self):
        print(self.img_files['x'][:5])
        print(self.img_files['y'][:5])

    def self_sanity_check(self):
        for path_gt, path_lq in zip(self.img_files['x'], self.img_files['y']):
            assert path_gt.replace('/clean/', f'/{self.noise_type}_noise_{self.noise_level}/') == path_lq


    def __getitem__(self, index):
        sample_index = _initilize_data_mode(self.data_mode)

        if self.data_mode['x']:
            img_gt = _im_read_resize(self.img_files['x'][index], self.WHC)
            filename_x = str(
                '/'.join(
                    self.img_files['x'][index].rsplit('/', self.keep_last_n_dirs)[1:]
                )
                )   

        # check the size of images.
        if (img_gt.ndim < 3) or (img_gt.shape[2] != self.WHC[2]):
            return None

        if self.data_mode['y']:
            img_lq = _im_read_resize(self.img_files['y'][index], self.WHC) 
            filename_y = str(
                '/'.join(
                    self.img_files['y'][index].rsplit('/', self.keep_last_n_dirs)[1:]
                )
                ) 

        if self.task_mode == 'DEN':       
            # sync to restormer
            if self.Training == True:
                # random crop
                img_gt, img_lq = paired_random_crop(
                    img_gt, img_lq, 
                    lq_patch_size=self.WHC[0], # take size of W
                    scale=1,
                    )

                # flip, rotation augmentations
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=True,
                                        float32=True)
        
        # print("*"*50)
        # print(filename_x)
        # print(filename_y)
        # print("*"*50)
        # print(filename_x.replace('/clean/', f'/{self.noise_type}_noise_{self.noise_level}/'))
        assert self.img_files['x'][index].replace('/clean/', f'/{self.noise_type}_noise_{self.noise_level}/') == self.img_files['y'][index]

        return {
            'x': img_gt,
            'y': img_lq,
            'filename_x': filename_x,
            'filename_y': filename_y
        }