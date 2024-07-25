import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
from dataset.dataset_cholecseg8k import *
from losses import DC_and_CE_loss, MemoryEfficientSoftDiceLoss
import utils
import math
from model import UNet,Uformer
import skimage
from skimage import img_as_float32, img_as_ubyte, io
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='Multi-label image segmentation')
parser.add_argument('--input_dir', default='./dataV3/CholecSeg8k/test/',
    type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='./logs/CholecSeg8k/Uformer_T_/results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./logs/CholecSeg8k/Uformer_T_/models/model_final.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_T', type=str, help='arch')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')    
parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
parser.add_argument('--segmentation', action='store_true', default=False, help="whether it's a segmentation problem or not")
parser.add_argument('--num_classes', type=int, default=2, help='number of classes for segmentation')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps_a', type=int, default=128, help='patch size a of training sample')
parser.add_argument('--train_ps_b', type=int, default=128, help='patch size b of training sample')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if args.save_images:
result_dir_img = os.path.join(args.result_dir, 'png')
utils.mkdir(result_dir_img)

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=False)

model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)
start_epoch = utils.load_start_epoch(args.weights) + 1 
train_losses = utils.load_train_losses(args.weights)
val_losses = utils.load_val_losses(args.weights)
mean_dice = utils.load_mean_dice(args.weights)
ema_dice = utils.load_ema_dice(args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

criterion = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': True, 'ddp': False, 'apply_nonlin': nn.Softmax(dim=1)}, {}, weight_ce=1, weight_dice=1, 
                                   dice_class=MemoryEfficientSoftDiceLoss).cuda()

total_test_loss = []
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader), 0):
        target = data[0].cuda()
        input_ = data[1].cuda()
        filenames = data[3]
        
        with torch.cuda.amp.autocast(): 
            prediction = model_restoration(input_)
            test_loss = criterion(prediction, target)
        total_test_loss.append(test_loss.detach().cpu().numpy())

        pred_probs = torch.softmax(prediction, 1)
        segmentations = pred_probs.argmax(1)

        for index, seg in enumerate(segmentations):
            num = filenames[index].split('-')[1]
            imag = seg.detach().cpu().numpy()
            io.imsave(f'{result_dir_img}/test_prediction-{num}', imag.astype(np.intc), check_contrast=False)
    
    final_loss = np.mean(total_test_loss)
    print('Test Loss: ', final_loss)
    torch.cuda.empty_cache()

        
    