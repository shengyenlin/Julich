from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import time
import datetime
import imghdr
from tqdm import tqdm

from model.denoiser_trainer import Denoiser_Trainer
from model.discriminator_trainer import Discriminator_Trainer
from model.adl_trainer import ADL_Trainer
from model import loss
from model.metric import MetricEval
from utils import util
from utils.dataloader_test import DataLoader_cls


dir_path = os.path.dirname(os.path.abspath(__file__))
print(f"Torch Version: {torch.__version__}")

# available device
num_gpus = torch.cuda.device_count()
#print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# root_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = "/mnt/big_disk/s.lin/Experiment_archives/ADL/"

parser = ArgumentParser()
parser.add_argument('--num-workers', type=int, default=8, action="store", help="number of channels")
parser.add_argument('--test-dirs', type=str, default='testCV', action="store", required=True, help="test directory")
parser.add_argument('--CHANNELS-NUM', type=int, default=-1, action="store", help="number of channels")
parser.add_argument('--EXPERIMENT', type=str, default='', action="store", help="name of experiment")
parser.add_argument('--json-file', type=str, action="store", default='', required=True, help="configuration file")
parser.add_argument('--save-images', choices=('True','False'), default='True', help="save the results as image")
parser.add_argument('--result_dir', type=str)
parser.add_argument('--weight_path', type=str)
args=parser.parse_args()



class Test(object):
    def __init__(self,
                device,
                num_workers, 
                args, 
                config, 
                eval_params):

        # Device config
        self.args = args
        self.args.save_images = util.boolean_string(self.args.save_images)

        self.device = device
        self.config = config['data']
        self.num_workers = num_workers
        self.eval_params = eval_params
        self.sigma = 1

    def __call__(self):
        load_save_device = self.config['localhost'] if self.config['localhost'] is not None else None

        output_dir = util.makedirs_fn(args.result_dir)

        # Load model =============
        ckpt_path = args.weight_path 

        model_name = 'efficient_Unet'

        # load model
        model = util.get_model(
            model_name, 
            self.args.CHANNELS_NUM, self.args.CHANNELS_NUM, 
            ckpt_path, device
            )



        # Load Data =============
        dataLoader_obj = DataLoader_cls(num_workers=self.num_workers,
                                        channels_num=self.args.CHANNELS_NUM,
                                        # clean_img_dir="/mnt/big_disk/s.lin/RestormerGrayScaleData/clean",
                                        test_ds_dir=self.args.test_dirs,
                                        config=self.config
                                    )
        ds_test_loader = dataLoader_obj()


        # apply model and get results
        model.eval()
        # out_noisy_img_dir = "/mnt/big_disk/s.lin/RestormerGrayScaleData/noisy_fot_ADL/test"
        with torch.no_grad():
            for i, batch_i in tqdm(enumerate(ds_test_loader)):
                extension = os.path.splitext(batch_i['filename'][0])[1]
                file_ = batch_i['filename'][0].split(extension)[0]

                inp = batch_i['y'].to(torch.float32).to(device)  # LQ
            
                ypred,_,_=model(
                    inp
                    )
                
                if self.args.save_images:
                    filename = file_.split('/')[-1]

                    out_dir = util.makedirs_fn(output_dir, '/'.join(file_.split('/')[1:-1]))
                    save_image(ypred, f'{out_dir}/{filename}.png')
        

if __name__== '__main__':
    
    # Read configuration file =============
    config = util.read_config(args.json_file)
    [print(key, item) for key, item in config.items()]

    eval_params = {
        'psnr': MetricEval.psnr,
        'ssim': MetricEval.ssim
    }

    Test = Test(device=device,
                    num_workers=args.num_workers,
                    args= args, 
                    config=config, 
                    eval_params = eval_params
                    )


    ticks = time.time()
    Test()
    elapsed_time = time.time() - ticks
    print(f"[i] Test took hh:mm:ss->{str(datetime.timedelta(seconds=elapsed_time))} (hh:mm:ss).") 
    print('Done!')