import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'y' 
        input_dir = 'x'
        
        segmentation_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.segmentation_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in segmentation_files if is_png_file(x)]
        self.input_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in input_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.segmentation_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        segmentation = torch.from_numpy(np.float32(load_img(self.segmentation_filenames[tar_index])))
        input = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
        
        segmentation = segmentation.permute(2,0,1)
        input = input.permute(2,0,1)

        segmentation_filename = os.path.split(self.segmentation_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = segmentation.shape[1]
        W = segmentation.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        segmentation = segmentation[:, r:r + ps, c:c + ps]
        input = input[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        segmentation = getattr(augment, apply_trans)(segmentation)
        input = getattr(augment, apply_trans)(input)        

        return segmentation, input, segmentation_filename, input_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'y'
        input_dir = 'x'
        
        segmentation_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.segmentation_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in segmentation_files if is_png_file(x)]
        self.input_filenames = [os.path.join(rgb_dir, input_dir, x) for x in input_files if is_png_file(x)]
        

        self.tar_size = len(self.segmentation_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        segmentation = torch.from_numpy(np.float32(load_img(self.segmentation_filenames[tar_index])))
        input = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
                
        segmentation_filename = os.path.split(self.segmentation_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]

        segmentation = segmentation.permute(2,0,1)
        input = input.permute(2,0,1)

        return segmentation, input, segmentation_filename, input_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)