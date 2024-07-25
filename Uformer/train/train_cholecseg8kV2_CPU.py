import os
import sys

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
from dataset.dataset_cholecseg8k import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

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

from losses import DC_and_CE_loss, MemoryEfficientSoftDiceLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
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
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration 
     

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

######### Loss ###########
criterion = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1, 
                                   dice_class=MemoryEfficientSoftDiceLoss)

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':[opt.train_ps_a, opt.train_ps_b]}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=False)

#img_options_val = {'patch_size':[opt.val_ps_a, opt.val_ps_b]}
val_dataset = get_validation_data(opt.val_dir)

val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### validation ###########
'''
with torch.no_grad():
    model_restoration.eval()
    dice_model_init = []
    count =0
    metric = MemoryEfficientSoftDiceLoss(batch_dice=True,
                                   smooth= 1e-5, do_bg= False, ddp= False)
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0]
        input_ = data_val[1]
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)  
        value_dice = metric(restored, target)
        if count == 0:
            prev_dice = value_dice
        total_dice = 0.9 * prev_dice + 0.1 *value_dice
        dice_model_init.append(total_dice)
        prev_dice = total_dice
        count +=1
    dice_model_init = sum(dice_model_init)/len_valset
    print('Initial EMA Pseudo Dice: %.4f'%(dice_model_init))
'''
######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_ema_dice = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()

        target = data[0]
        input_ = data[1]

        with torch.cuda.amp.autocast():  ### Cambiar aquí ###
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()

        #### Evaluation ####
        if (i+1)%eval_now==0 and i>0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_restoration.eval()
                dice_val_rgb = []
                count =0
                metric = MemoryEfficientSoftDiceLoss(batch_dice=True,
                                   smooth= 1e-5, do_bg= False, ddp= False)
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0]
                    input_ = data_val[1]
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1)  
                    value_dice = metric(restored, target)
                    if count == 0:
                        prev_dice = value_dice
                    total_dice = 0.9 * prev_dice + 0.1 *value_dice
                    dice_val_rgb.append(total_dice)
                    prev_dice = total_dice
                    count +=1

                dice_val_rgb = sum(dice_val_rgb)/len_valset
                
                if dice_val_rgb > best_ema_dice:
                    best_ema_dice = dice_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Epoch %d iteration %d\t EMA Pseudo Dice CholecSeg8k: %.4f\t] ----  [best_Ep_CholecSeg8k %d best_it_CholecSeg8k %d best_ema_dice_CholecSeg8k %.4f] " % (epoch, i, dice_val_rgb,best_epoch,best_iter,best_ema_dice))
                with open(logname,'a') as f:
                    f.write("[Epoch %d iteration %d\t EMA Pseudo Dice CholecSeg8k: %.4f\t] ----  [best_Ep_CholecSeg8k %d best_it_CholecSeg8k %d best_ema_dice_CholecSeg8k %.4f] " \
                        % (epoch, i, dice_val_rgb,best_epoch,best_iter,best_ema_dice)+'\n')
                model_restoration.train()
        torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
