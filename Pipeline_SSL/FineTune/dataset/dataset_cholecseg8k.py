import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderCholec(Dataset):
    def __init__(self, rgb_dir, mean_std):
        super(DataLoaderCholec, self).__init__()

        gt_dir = 'y' 
        input_dir = 'x'

        segmentation_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.segmentation_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in segmentation_files if is_png_file(x)]
        self.input_filenames = [os.path.join(rgb_dir, input_dir, x) for x in input_files if is_png_file(x)]

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
        segmentation = torch.from_numpy(np.float32(load_img(self.segmentation_filenames[tar_index])))

        img = self.to_tensor(img)

        segmentation.unsqueeze(0)
        
        segmentation_filename = os.path.split(self.segmentation_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]

        return segmentation, img, segmentation_filename, input_filename

##################################################################################################


def get_data(rgb_dir, mean_std):
    assert os.path.exists(rgb_dir)
    return DataLoaderCholec(rgb_dir, mean_std)
