import argparse
import os
import random
import time
import numpy as np
import skimage
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
from skimage import io
import options
import utils
from utils.poly_lr import PolyLRScheduler
from utils.model_utils import *
import datetime

parser = argparse.ArgumentParser(description='Multi-label image segmentation')
parser.add_argument('--input_dir', default='./Parte2/target_data/test/',
    type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='./logs/CholecSeg8k/results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./logs/CholecSeg8k/models/model_latest.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes for segmentation')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='contrastive model architecture backbone.')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    # if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'png')
    utils.mkdir(result_dir_img)

    cholecSeg8k_mean_std = [[0.337, 0.212, 0.181],[0.283, 0.223, 0.193]]
    test_dataset = get_data(args.input_dir, cholecSeg8k_mean_std)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    contrastive = FeatureGetter(backbone=args.arch)
    generative = SimpleViT(
        image_size = (16,32),
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 512
    )
    
    model = EndToEnd(contrastive= contrastive, generative= generative, num_classes=args.num_classes)

    utils.load_checkpoint(model,args.weights)

    print("===>Testing using weights: ", args.weights)

    model.cuda()
    model.eval()

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
                prediction = model(input_)
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

        
if __name__ == '__main__':
    main()
