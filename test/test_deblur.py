import os.path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from collections import OrderedDict
import math
from model import Uformer

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as ssim_loss

from tqdm import tqdm


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def to_numpy(tensor):   # tensor: (1, 1, h, w)的张量，像素值已归一化
    tensor_cpu = tensor.cpu()
    output_numpy = tensor_cpu.numpy()
    output_numpy = output_numpy.squeeze()
    img_255 = img_as_ubyte(output_numpy)    # 将浮点数 [0, 1] 范围的 NumPy 图像数组转换回 [0, 255] 范围的 uint8 整数类型
    return img_255

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# NOTE: 模型初始化和加载
model = Uformer(img_size=128, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=1)
weight_path = './../weight/model_best.pth'
load_checkpoint(model, weight_path)
model.to(device)
model.eval()


# NOTE: 确保下面input_path和target_path目录里面的图片是512×512大小才能正常输入到网络
input_path = './input'
target_path = './target'
output_path = './output'
input_filenames = os.listdir(input_path)
target_filenames = os.listdir(target_path)

PSNR = []
SSIM = []
for i, (input_name, target_name) in enumerate(tqdm(zip(input_filenames, target_filenames), desc="Testing Progress"), start=1):
    # NOTE: 输入图像预处理
    input_numpy = cv2.imread(os.path.join(input_path, input_name))
    input_gray = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2GRAY)
    input_hw1 = np.expand_dims(input_gray, axis=2)
    input_tensor_1hw = F.to_tensor(input_hw1)
    input_tensor_11hw = input_tensor_1hw.unsqueeze(0)
    input_tensor = input_tensor_11hw.to(device)
    # NOTE: 推理
    with torch.no_grad():
        output = model(input_tensor)
    tensor_01 = torch.clamp(output, 0, 1)
    numpy_255_gray = to_numpy(tensor_01)
    # cv2.imwrite(os.path.join(output_path, input_name), numpy_255_gray)  # (h,w)的ndarray数组就表示灰度图,可以直接用cv2.imwrite保存为png图片
    # NOTE: 读取目标图像(ground truth)并计算指标
    target = cv2.imread(os.path.join(target_path, target_name))
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    psnr = psnr_loss(numpy_255_gray, target_gray)
    ssim = ssim_loss(numpy_255_gray / 255., target_gray / 255.)
    PSNR.append(psnr)
    SSIM.append(ssim)

psnr_avg = sum(PSNR) / len(PSNR)
ssim_avg = sum(SSIM) / len(SSIM)
print(f'PSNR: {psnr_avg}\t\tSSIM: {ssim_avg}')
# PSNR: 26.903118556109916		SSIM: 0.8709062059987996

