import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# 生成一个运动模糊核。
def generate_motion_blur_kernel(size, angle, motion_type='linear'):
    kernel = np.zeros((size, size))
    # 将角度转换为弧度
    rad_angle = np.deg2rad(angle)
    # 计算运动的中心点
    center = size // 2
    if motion_type == 'linear':
        # 在核的中心画一条线来模拟线性运动
        x1 = int(center - (size - 1) / 2 * np.cos(rad_angle))
        y1 = int(center - (size - 1) / 2 * np.sin(rad_angle))
        x2 = int(center + (size - 1) / 2 * np.cos(rad_angle))
        y2 = int(center + (size - 1) / 2 * np.sin(rad_angle))
        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    # 归一化核，使其所有元素之和为1
    kernel /= kernel.sum()
    return kernel

# 将运动模糊核应用到图像上。
def apply_motion_blur(image, kernel):
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

# 批量处理一个目录下的所有图片，生成运动模糊版本
def process_images_in_directory(input_dir, output_dir, blur_config):
    os.makedirs(output_dir, exist_ok=True)
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    print(f" {len(image_files)} 张图片，开始处理")
    for filename in tqdm(image_files, desc="Processing Images"):
        # 构建完整的文件路径
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        # 读取清晰图像
        clear_image = cv2.imread(input_path)
        if clear_image is None:
            print(f"无法读取图片 {input_path}，已跳过。")
            continue

        # --- 随机化模糊参数 ---
        # 从配置范围中随机选择一个模糊程度
        blur_size = random.randint(blur_config['min_size'], blur_config['max_size'])
        # 随机选择一个模糊角度
        blur_angle = random.randint(blur_config['min_angle'], blur_config['max_angle'])
        # 生成运动模糊核
        motion_kernel = generate_motion_blur_kernel(blur_size, blur_angle)
        # 应用模糊
        blurred_image = apply_motion_blur(clear_image, motion_kernel)
        # 保存模糊后的图像
        cv2.imwrite(output_path, blurred_image)
    print("所有图片处理完成")
    print(f"模糊图片已保存到: {output_dir}")


if __name__ == '__main__':
    # --- 设置目录 ---
    # 存放清晰图片的目录 (groundtruth)
    INPUT_IMAGE_DIR = './../test/target'  # 不能有中文，所以使用相对路径
    # 想把生成的模糊图片存放在哪里 (input)
    OUTPUT_IMAGE_DIR = './../test/input'

    # --- 配置模糊参数 ---
    # 可以调整这些值来改变模糊的效果
    blur_parameters = {
        'min_size': 5,  # 最小模糊程度 (核尺寸)
        'max_size': 10,  # 最大模糊程度
        'min_angle': 0,  # 最小模糊角度
        'max_angle': 360  # 最大模糊角度
    }
    # 执行批量处理
    process_images_in_directory(INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR, blur_parameters)
