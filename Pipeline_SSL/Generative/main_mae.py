
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import skimage
import logging
from skimage import io
from tqdm import tqdm
from models.simple_vit import SimpleViT
from models.mae import MAE
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='MAE HyperKvir(24k) training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--work_dir',  type=str, default='./workdir/',
                    help='name of data list file')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
args = parser.parse_args()

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('USNet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

saved_path = args.work_dir
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, 'train_new.log'))

def is_pt_file(filename):
    return any(filename.endswith(extension) for extension in [".pt"])

class DataLoaderFeatureMaps(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderFeatureMaps, self).__init__()

        input_files = sorted(os.listdir(rgb_dir))

        self.input_filenames = [os.path.join(rgb_dir, x) for x in input_files if is_pt_file(x)]

        self.tar_size = len(self.input_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        img_path = self.input_filenames[tar_index]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]
        with h5py.File(img_path, 'r') as hf:
          img = torch.tensor(hf[input_filename][:])
        return img, input_filename


def get_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderFeatureMaps(rgb_dir)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def main():

    if args.gpu is not None:
        print("Using GPU: {} for training".format(args.gpu))

    # create model
    print("=> Creating model 'MAE'")
    v = SimpleViT(
        image_size = (16,32),
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 512
    )
    v.cuda()
    model = MAE(
        encoder = v,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 4,      # anywhere from 1 to 8
        decoder_heads = 4
    )
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("No gpu selected.")
    
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, betas=(0.9, 0.95))

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Checkpoint '{}' loaded (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    traindir = args.data

    train_dataset = get_data(os.path.join(traindir, 'train'))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=False)
    val_dataset = get_data(os.path.join(traindir, 'val'))
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=False)
    test_dataset = get_data(os.path.join(traindir, 'test'))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=False)
    

    for epoch in range(args.start_epoch, args.epochs):
        
        logger.info('Epoch: {}'.format(epoch))

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        val(val_loader, model, epoch, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename=os.path.join(saved_path, 'checkpoint_latest_new.pth.tar'.format(epoch)))

        torch.save(v.state_dict(), os.path.join(saved_path, 'checkpoint_ViT_latest_new.pth.tar'))
    
    test(test_loader, model, epoch, args)
        

def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    
    if epoch == 0:
        progress.display(0)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            _input = images.cuda(args.gpu, non_blocking=True)

        loss = model(_input)      
        losses.update(loss.item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i)

def val(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Val Loss', ':.4f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, _) in enumerate(tqdm(val_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            _input = images.cuda(args.gpu, non_blocking=True)

        loss = model(_input)      
        losses.update(loss.item(), images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    progress.display(i)

def test(test_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Test Loss', ':.4f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, _) in enumerate(tqdm(test_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            _input = images.cuda(args.gpu, non_blocking=True)

        loss = model(_input)      
        losses.update(loss.item(), images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    progress.display(i)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()