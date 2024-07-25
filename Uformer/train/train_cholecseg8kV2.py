import os
import sys
import numpy as np
import skimage
from skimage import io

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Multi-label image segmentation')).parse_args()
print(opt)

import utils
from utils.poly_lr import PolyLRScheduler
from dataset.dataset_cholecseg8k import *
from utils.model_utils import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import DC_and_CE_loss, MemoryEfficientSoftDiceLoss, RobustCrossEntropyLoss, SoftDiceLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, PolynomialLR
from timm.utils import NativeScaler




######### Logs dir ###########
log_dir = os.path.join(opt.save_dir,opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 0
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model_restoration.parameters(), opt.lr_initial, weight_decay=opt.weight_decay,
                                    momentum=0.99, nesterov=True)
else:
    raise Exception("Error optimizer...")
train_losses = []
val_losses = []
mean_dice = []
ema_dice = []

######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration.cuda() 
     

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    '''step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()'''
    print('\nUsing Poly LR!\n')
    scheduler = PolyLRScheduler(optimizer, opt.lr_initial, opt.nepoch)
    scheduler.step(0)

######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest)
    train_losses = utils.load_train_losses(path_chk_rest)
    val_losses = utils.load_val_losses(path_chk_rest)
    mean_dice = utils.load_mean_dice(path_chk_rest)
    ema_dice = utils.load_ema_dice(path_chk_rest)

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(0, start_epoch+1):
        scheduler.step(i)
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

######### Loss ###########
    
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn
criterion = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': True, 'ddp': False, 'apply_nonlin': nn.Softmax(dim=1)}, {}, weight_ce=1, weight_dice=1, 
                                   dice_class=MemoryEfficientSoftDiceLoss).cuda()


######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':[opt.train_ps_a, opt.train_ps_b]}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)

val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch-1))
best_ema_dice = 0
best_epoch = 0
best_iter = 0

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
count =0
metric = MemoryEfficientSoftDiceLoss(batch_dice=True,
                                   smooth= 1e-5, do_bg= True, ddp= False,apply_nonlin= nn.Softmax(dim=1), metric= True).cuda()



for epoch in range(start_epoch, opt.nepoch):
    current_lr = scheduler.get_lr()[0]
    print("------------------------------------------------------------------")
    print("Epoch: {}\tLearningRate: {:.6f}".format(epoch, current_lr))
    print("------------------------------------------------------------------")
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_loss_val = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()

        with torch.cuda.amp.autocast():  
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters(), clip_grad=12)
        epoch_loss +=loss.item()

        #### Evaluation ####
    with torch.no_grad():
        model_restoration.eval()
        if count == 0:
            prev_dice = 0.0
        else:
            prev_dice = dice_val_rgb_ab
                
        to_print = []
        dice_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
                    
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            filenames = data_val[2]
            with torch.cuda.amp.autocast():
                restored = model_restoration(input_)
                val_loss = criterion(restored, target)
            epoch_loss_val +=val_loss.item()

            value_dice, values_to_print = metric(restored, target)
            to_print.append(values_to_print)
            dice_val_rgb.append(value_dice)

        epoch_loss_val = opt.batch_size * epoch_loss_val /len_valset
        to_print = (sum(to_print) / len_valset) * opt.batch_size
        dice_value = (sum(dice_val_rgb)/len_valset) * opt.batch_size
        if count == 0:
            dice_val_rgb_ab = dice_value
        else: 
            dice_val_rgb_ab = prev_dice * 0.9 + dice_value * 0.1

        if dice_val_rgb_ab > best_ema_dice:
            best_ema_dice = dice_val_rgb_ab
            best_epoch = epoch
            best_iter = i
            print('\nPseudo Dice: {}\nYayy! New best EMA Pseudo Dice: {:.4f}'.format(torch.round(to_print, decimals=4), dice_val_rgb_ab)) 

        else:
            print('\nPseudo Dice: {}\nEMA Pseudo Dice: {:.4f}'.format(torch.round(to_print, decimals=4), dice_val_rgb_ab)) 
        with open(logname,'a') as f:
            f.write("[Epoch %d\t EMA Pseudo Dice CholecSeg8k: %.4f\t] ----  [best_Ep_CholecSeg8k %d best_it_CholecSeg8k %d best_ema_dice_CholecSeg8k %.4f] " \
                % (epoch, dice_val_rgb_ab,best_epoch,best_iter,best_ema_dice)+'\n')
        
        dice_value_cpu = dice_value.cpu()
        dice_val_rgb_ab_cpu = dice_val_rgb_ab.cpu()
        mean_dice.append(dice_value_cpu)
        ema_dice.append(dice_val_rgb_ab_cpu)

        model_restoration.train()
        torch.cuda.empty_cache()
        count +=1
    scheduler.step(epoch+1)
    epoch_loss = opt.batch_size * epoch_loss /len_trainset

    train_losses.append(epoch_loss)
    val_losses.append(epoch_loss_val)

    epoch_end_time = time.time()
    print("Epoch Loss: {:.4f}\t Duration: {:.4f}".format(epoch_loss, epoch_end_time-epoch_start_time))

    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, epoch_end_time-epoch_start_time,epoch_loss, current_lr)+'\n')

    plot_progress_png(model_dir, epoch, train_losses, val_losses, mean_dice, ema_dice)

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'mean_dice': mean_dice,
                'ema_dice': ema_dice
                }, os.path.join(model_dir,"model_latest.pth"))   
