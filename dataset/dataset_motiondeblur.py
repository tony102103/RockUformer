import numpy as np
import os

import torchvision
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob


import cv2

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'groundtruth' 
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        # clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        # 读取灰度图(含中文路径使用np.fromfile和cv2.imdecode)
        clean_gray = cv2.imdecode(np.fromfile(self.clean_filenames[tar_index], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        noisy_gray = cv2.imdecode(np.fromfile(self.noisy_filenames[tar_index], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # (H, W) 的 numpy 数组转换为 (H, W, 1)
        clean = np.expand_dims(clean_gray, axis=2)
        noisy = np.expand_dims(noisy_gray, axis=2)
        # 转为tensor、归一化
        clean = torch.from_numpy(clean).float().permute(2, 0, 1) / 255.0  # (H,W,1) -> (1,H,W)
        noisy = torch.from_numpy(noisy).float().permute(2, 0, 1) / 255.0  # (H,W,1) -> (1,H,W)
        # 文件名
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        # 随机裁剪
        ps = self.img_options['patch_size']     # 128
        _, H, W = clean.shape
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        # 随机数据增强
        apply_trans = transforms_aug[random.getrandbits(3)]
        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################

class DataLoaderVal_deblur(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal_deblur, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_png_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'groundtruth', x) for x in tar_files if is_png_file(x)]

        self.img_options = img_options
        self.ps = self.img_options['patch_size'] if img_options is not None else None

        self.tar_size = len(self.tar_filenames)
        # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        index_ = index % self.tar_size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        # inp_img = Image.open(inp_path)
        # tar_img = Image.open(tar_path)
        # 读取灰度图
        inp_img = Image.open(inp_path).convert('L')  # 'L' 代表 8-bit 灰度
        tar_img = Image.open(tar_path).convert('L')

        # 中心区域裁剪
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        # TF.to_tensor 会自动处理灰度图，将 (H, W) 的 PIL 灰度图转换为 (1, H, W) 的张量
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)


        return clean, noisy, clean_filename, noisy_filename

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


##################################################################################################
def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_deblur_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_deblur(rgb_dir, img_options, None)
def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)



