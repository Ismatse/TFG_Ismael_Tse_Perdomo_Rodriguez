import argparse
import os
import h5py
from tqdm import tqdm
from skimage import io

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.models as models
from torchvision.datasets.folder import default_loader
from models.feature_getter import FeatureGetter
import torchvision.transforms as T
from setlogger import get_logger

torch.set_printoptions(sci_mode=False)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch HyperKvir Feature Extraction')

parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--checkpoint-dir', default='', type=str, metavar='PATH',
                    help='path to checkpoint from which to extract features (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--work-dir', default='', type=str, metavar='PATH',
                    help='directory to store feature_maps (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
args = parser.parse_args()
saved_path = args.work_dir

if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, 'get_feature_maps.log'))

def is_jpg_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, mean_std):
        super(DataLoaderVal, self).__init__()

        input_files = sorted(os.listdir(rgb_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.input_filenames = [os.path.join(rgb_dir, x) for x in input_files if is_jpg_file(x)]

        self.tar_size = len(self.input_filenames)

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        img_path = self.input_filenames[tar_index]
        img = default_loader(img_path)
        image = self.to_tensor(img)

        input_filename = os.path.split(self.input_filenames[tar_index])[-1]

        return image, input_filename

def get_val_data(rgb_dir, mean_std):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, mean_std)

def main():
    main_worker(args)

def main_worker(args):
    if args.gpu is not None:
        print("Using GPU: {} for training".format(args.gpu))
    
    model = FeatureGetter(backbone=args.arch)
    print(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    if os.path.isfile(args.checkpoint_dir):
        print("=> Loading checkpoint '{}'".format(args.checkpoint_dir))
        checkpoint = torch.load(args.checkpoint_dir)
        features = [x for x in checkpoint['state_dict'].keys() if x.split('.')[0]=='features']
        features_keys = [x for x in features if not (x.startswith('features.7.2.conv3') or x.startswith('features.7.2.bn3'))]
        state_dict_filtered = {k: v for k, v in checkpoint['state_dict'].items() if k in features_keys}

        model.load_state_dict(state_dict_filtered)
        print("=> Checkpoint '{}' loaded (epoch {})"
                .format(args.checkpoint_dir, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> No checkpoint found at '{}'".format(args.checkpoint_dir))

    cudnn.benchmark = True

    traindir = args.data
    hyperKvir_mean_std = [[0.486, 0.317, 0.258],[0.333, 0.248, 0.226]]

    val_dataset = get_val_data(traindir, hyperKvir_mean_std)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    logger.info('Feature maps Extraction:')
    
    model.eval()
    counter = 1
    for _,(images, img_file_name) in enumerate(tqdm(val_loader)):
        imagenes = images.cuda(args.gpu, non_blocking=True)
        feature_maps = model(imagenes)
        feature_maps = feature_maps.detach().cpu()
        for nombre_archivo, feature_map in zip(img_file_name, feature_maps):
            numero = int(nombre_archivo.split("_")[1].split(".")[0])
            nombre_archivo_feature_map = f"feature_map_{numero:05d}.pt"
            print(counter)
            ruta_guardado = os.path.join(saved_path, 'feature_maps_light',nombre_archivo_feature_map)
            with h5py.File(ruta_guardado, 'w') as hf:
                hf.create_dataset(nombre_archivo_feature_map, data=feature_map.numpy())
            counter += 1
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
    