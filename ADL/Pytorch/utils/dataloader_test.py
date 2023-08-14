from typing import Any, Callable, Dict, Tuple, Union, Iterable
import os
import numpy as np
import random
import re

import cv2

import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
from skimage import io, color, transform
import warnings
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler



from utils.transform_collections import img2tensor


class DataLoader_cls(Dataset): 
    def __init__(self,
                num_workers: int,
                channels_num: int,
                # clean_img_dir,
                test_ds_dir: Union[str,list],
                config:  Union[str,list]
        )->Union[DataLoader, DataLoader, DataLoader]: 
        r"""Data loader using DDP
        
        Args:
            num_workers: number of workers. It will be divided by the number of gpus
            channels_num: input channels (RGB:3, grey:1)
            train_ds_dir: A list of directories for training datasets
            config: configuration file for data
        """

        
        # chech the directories
        self.test_dir = _get_dir(test_ds_dir)
        # self.clean_img_dir = _get_dir(clean_img_dir)

        # dataloader params
        self.config = config
        self.DL_params = {'batch_size': 1,
                        'num_workers': num_workers,
                    }

        # x: ground-truth, y: noisy sample (if False, we will add synthesized noise to x)
        self.data_mode = {'y':True, 'x':False, 'mask':False, 'filename':False}
        self.DS_params = {'data_mode': self.data_mode, 
                        'task_mode': config['task_mode'] ,
                        'WHC': [config['W'],config['H'],channels_num], 
                        'img_format': config['img_types'],
                    }

    def __call__(self):

        # create dataloader for teste
        img_files = _get_files(
            self.test_dir, 
            # self.clean_img_dir,
            self.config['img_types'], 
            self.data_mode
            )
        Dataset_Test  = Dataset_cls(img_files=img_files,  
                                    Training=False, 
                                    noise_level=0, 
                                    **self.DS_params)
        test_loader  = DataLoader(dataset= Dataset_Test, shuffle= False, 
                                    collate_fn=collate_fn, **self.DL_params)

        return test_loader


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

def _get_files(
        dirs_:Union[list,Iterable[str]], 
        # clean_img_dir, 
        img_format, data_mode):
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

    # img_dirs['x'] = [os.path.join(path, name) 
    #                 for dir_i in clean_img_dir 
    #                     for path, subdirs, files_ in os.walk(dir_i)  
    #                         for name in files_ 
    #                             if name.lower().endswith(tuple(img_format))]
    
    img_dirs['y'] = [os.path.join(path, name) 
                    for dir_i in dirs_ 
                        for path, subdirs, files_ in os.walk(dir_i)  
                            for name in files_ 
                                if name.lower().endswith(tuple(img_format))]
    # if data_mode['y']:
    #     img_dirs['y'] =  [
    #         files.replace('/clean/', f'/noise_{noise_level}/') for files in img_dirs['x']
    #         ]

    # if data_mode['mask']:
    #     img_dirs['mask'] =  [files.replace('/HR/', '/mask/') for files in x_files]

    return img_dirs





# sync with testing
def _im_read_resize(PATH:str, WHC:Tuple[int, Iterable[int]]):
    """read and resize images"""

    assert isinstance(WHC, (list, tuple)) and len(WHC)==3, "Invalid tuple for width, height, channel"
    Width, Height, Channel = WHC[0], WHC[1], WHC[2]

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

    # # img.shape[-1] == 3, only for rgb images
    # # testing set of BSD doesn't contain channel dimension
    # if Channel == 1 and img.shape[-1] == 3:
    #     img = color.rgb2gray(img)   

    # if img.ndim == 2:
    #     img = np.expand_dims(img, axis=-1) 

    # # Deprecated during testing
    # if Width > 0 and Height > 0:
    #     print(f"img shape before: {img.shape}")  
    #     img = transform.resize(img, (Height, Width)) 
    #     print(f"img shape after: {img.shape}")

    # # print(f"img shape: {img.shape}") -> (W, H, 1)

    # return np.array(img, dtype=np.float64)  




class Dataset_cls(Dataset):
    def __init__(self,
                img_files: str,
                Training: bool,
                data_mode:Dict[str, bool],
                task_mode:str,
                noise_level: Union[list,float],
                WHC:list,
                img_format:list,
                keep_last_n_dirs:int=3
                ):
        r""" Dataset class for one dataset

        Args:
            img_files: the filenames of images
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
        self.noise_level = 0
        self.WHC = WHC
        self.Training = Training
        self.img_files = img_files 
        self.keep_last_n_dirs = keep_last_n_dirs
        
    def __len__(self):
        return len(self.img_files['y'])


    def __getitem__(self, index):
        sample_index = _initilize_data_mode(self.data_mode)

        if self.data_mode['x']:
            img_gt = _im_read_resize(self.img_files['x'][index], self.WHC)
            # sample_index['filename'] = str('/'.join(
            #                 self.img_files['x'][index].rsplit('/', self.keep_last_n_dirs)[1:]
            #                 ))

        if self.data_mode['y']:
            img_lq = _im_read_resize(self.img_files['y'][index], self.WHC)
            filename = str('/'.join(
                            self.img_files['y'][index].rsplit('/', self.keep_last_n_dirs)[1:]
                            ))    

        if self.task_mode == 'GTandLQ':
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=True,
                                        float32=True)
            
        elif self.task_mode == 'LQonly':
            img_lq = img2tensor(img_lq,
                                bgr2rgb=True,
                                float32=True
                                )  
            # placeholder
            img_gt = img_lq

        return {
            'x': img_gt,
            'y': img_lq,
            'filename': filename
        }