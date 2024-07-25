import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm
from typing import List
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from models.feature_getter import FeatureGetter
from models.simple_vit import SimpleViT
from models.end_to_end import EndToEnd
from losses import DC_and_CE_loss, MemoryEfficientSoftDiceLoss
from setlogger import get_logger
from dataset.dataset_cholecseg8k import *
import options
import utils
from utils.poly_lr import PolyLRScheduler
from utils.model_utils import *
import datetime
from timm.utils import NativeScaler

######### parser ###########
args = options.Options().init(argparse.ArgumentParser(description='Multi-label image segmentation')).parse_args()
print(args)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
torch.backends.cudnn.benchmark = True
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######### Logs dir ###########
log_dir = os.path.join(args.save_dir,args.dataset)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = get_logger(os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')) 
print("Now time is : ",datetime.datetime.now().isoformat())
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    if args.gpu is not None:
        logger.info("Using GPU: {} for training".format(args.gpu))
    
    contrastive = FeatureGetter(backbone=args.arch)
    logger.info('\nContrastive Model:\n',contrastive)

    generative = SimpleViT(
        image_size = (16,32),
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 512
    )
    logger.info('\nGenerative Model:\n',generative)
    
    model = EndToEnd(contrastive= contrastive, generative= generative, num_classes=args.num_classes)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = PolyLRScheduler(optimizer, args.lr, args.epochs)
    scheduler.step(0)
    start_epoch = 0
    train_losses = []
    val_losses = []
    mean_dice = []
    ema_dice = []

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("No gpu selected.")
    
    if args.resume:
        path_chk_rest = args.resume 
        print("Resume from "+path_chk_rest)
        utils.load_checkpoint(model,path_chk_rest) 
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
    else:
        if os.path.isfile(args.contrastive_weights):
            logger.info("=> Loading contrastive pre-training weights'{}'".format(args.contrastive_weights))
            checkpoint = torch.load(args.contrastive_weights)
            features = [x for x in checkpoint['state_dict'].keys() if x.split('.')[0]=='features']
            features_keys = [x for x in features if not (x.startswith('features.7.2.conv3') or x.startswith('features.7.2.bn3'))]
            state_dict_filtered = {k: v for k, v in checkpoint['state_dict'].items() if k in features_keys}

            contrastive.load_state_dict(state_dict_filtered)
            logger.info("=> Contrastive pre-training weights loaded")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            logger.info("=> No contrastive pre-training weights found at '{}'".format(args.contrastive_weights))

        if os.path.isfile(args.generative_weights):
            logger.info("=> Loading generative pre-training weights '{}'".format(args.generative_weights))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.generative_weights, map_location=loc)
            generative.load_state_dict(checkpoint)
            logger.info("=> Generative pre-training weights loaded")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            logger.info("=> No generative pre-training weights found at '{}'".format(args.generative_weights))
    
    ######### Loss ###########
    def collate_outputs(outputs: List[dict]):
        """
        used to collate default train_step and validation_step outputs. If you want something different then you gotta
        extend this

        we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
        """
        collated = {}
        for k in outputs[0].keys():
            if np.isscalar(outputs[0][k]):
                collated[k] = [o[k] for o in outputs]
            elif isinstance(outputs[0][k], np.ndarray):
                collated[k] = np.vstack([o[k][None] for o in outputs])
            elif isinstance(outputs[0][k], list):
                collated[k] = [item for o in outputs for item in o[k]]
            else:
                raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                                f'Modify collate_outputs to add this functionality')
        return collated  

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

    def validation_step(batch: dict) -> dict:
            data = batch[1].cuda()
            target = batch[0].cuda()

            data = data.to(device, non_blocking=True)
            if isinstance(target, list):
                target = [i.to(device, non_blocking=True) for i in target]
            else:
                target = target.to(device, non_blocking=True)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.cuda.amp.autocast():
                output = model(data)
                del data
                l = criterion(output, target)

            # the following is needed for online evaluation. Fake dice (green line)
            axes = [0] + list(range(2, output.ndim))

            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

            mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

            return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_validation_epoch_end(val_outputs: List[dict], current_epoch, prev_dice, best_ema_dice, best_epoch, best_iter, i, mean_dice, ema_dice, val_losses):
            outputs_collated = collate_outputs(val_outputs)
            tp = np.sum(outputs_collated['tp_hard'], 0)
            fp = np.sum(outputs_collated['fp_hard'], 0)
            fn = np.sum(outputs_collated['fn_hard'], 0)

            loss_here = np.mean(outputs_collated['loss'])

            global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
            mean_fg_dice = np.nanmean(global_dc_per_class)

            if current_epoch == 0:
                dice_val_rgb_ab = mean_fg_dice
            else: 
                dice_val_rgb_ab = prev_dice * 0.9 + mean_fg_dice * 0.1
            
            if dice_val_rgb_ab > best_ema_dice:
                best_ema_dice = dice_val_rgb_ab
                best_epoch = current_epoch
                best_iter = i
                logger.info('\nYayy! New best EMA Pseudo Dice') 

            logger.info("[Epoch %d\t EMA Pseudo Dice CholecSeg8k: %.4f\t] ----  [best_Ep_CholecSeg8k %d best_it_CholecSeg8k %d best_ema_dice_CholecSeg8k %.4f] " \
                    % (current_epoch, dice_val_rgb_ab,best_epoch,best_iter,best_ema_dice)+'\n')
            
            mean_dice.append(mean_fg_dice)
            ema_dice.append(dice_val_rgb_ab)
            val_losses.append(loss_here)

            return best_ema_dice, best_epoch, best_iter
        
    criterion = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': True, 'ddp': False, 'apply_nonlin': nn.Softmax(dim=1)}, {}, weight_ce=1, weight_dice=1, 
                                   dice_class=MemoryEfficientSoftDiceLoss).cuda()
    
    
    cudnn.benchmark = True

    traindir = args.train_dir
    valdir = args.val_dir

    cholecSeg8k_mean_std = [[0.337, 0.212, 0.181],[0.283, 0.223, 0.193]]

    train_dataset = get_data(traindir, cholecSeg8k_mean_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    val_dataset = get_data(valdir, cholecSeg8k_mean_std)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    len_trainset = train_dataset.__len__()

    ######### train ###########
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,args.epochs-1))
    best_ema_dice = 0
    best_epoch = 0
    best_iter = 0

    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()
    
    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_lr()[0]
        print("------------------------------------------------------------------")
        print("Epoch: {}\tLearningRate: {:.6f}".format(epoch, current_lr))
        print("------------------------------------------------------------------")
        epoch_start_time = time.time()
        epoch_loss = 0

        for i, data in enumerate(tqdm(train_loader), 0): 
            # zero_grad
            optimizer.zero_grad()
            target = data[0].cuda()
            input_ = data[1].cuda()

            with torch.cuda.amp.autocast():  
                restored = model(input_)
                loss = criterion(restored, target)
            loss_scaler(loss, optimizer,parameters=model.parameters(), clip_grad=12)
            epoch_loss +=loss.item()

            #### Evaluation ####
        with torch.no_grad():
            model.eval()
            if epoch == 0:
                prev_dice = 0.0
            else:
                prev_dice = ema_dice[-1]
            val_outputs = []
            for _, data_val in enumerate((val_loader), 0):
                val_outputs.append(validation_step(data_val))

            best_ema_dice, best_epoch, best_iter = on_validation_epoch_end(val_outputs, epoch, prev_dice, best_ema_dice, best_epoch,
                                                                                            best_iter, i, mean_dice, ema_dice, val_losses)

            model.train()
            torch.cuda.empty_cache()
        scheduler.step(epoch+1)
        epoch_loss = args.batch_size * epoch_loss /len_trainset

        train_losses.append(epoch_loss)

        epoch_end_time = time.time()

        logger.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, epoch_end_time-epoch_start_time,epoch_loss, current_lr)+'\n')

        plot_progress_png(model_dir, epoch, train_losses, val_losses, mean_dice, ema_dice)

        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'mean_dice': mean_dice,
                    'ema_dice': ema_dice
                    }, os.path.join(model_dir,"model_latest.pth"))


    
if __name__ == '__main__':
    main()

