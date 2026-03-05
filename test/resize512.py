import cv2
import os
from tqdm import tqdm


def resize_images_in_directory(input_dir, output_dir, target_width, target_height):
    os.makedirs(output_dir, exist_ok=True)
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    if not image_files:
        print(f"在目录 '{input_dir}' 中没有找到支持的图片文件。")
        return
    print(f"找到了 {len(image_files)} 张图片，开始进行resize处理...")
    for filename in tqdm(image_files, desc="Resizing Images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"警告: 无法读取图片 {input_path}，已跳过。")
                continue
            target_size = (target_width, target_height)
            if image.shape[1] > target_width or image.shape[0] > target_height:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC

            resized_image = cv2.resize(image, target_size, interpolation=interpolation)
            cv2.imwrite(output_path, resized_image)
        except Exception as e:
            print(f"处理图片 {input_path} 时发生错误: {e}")
    print("所有图片处理完成！")
    print(f"Resize后的图片已保存到: {output_dir}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 1. 设置你的目录 ---
    # 包含你想要resize的图片的目录
    INPUT_DIR = './target'
    # 你想把resize后的图片存放在哪里
    OUTPUT_DIR = './target'

    # --- 2. 设置目标尺寸 ---
    TARGET_WIDTH = 512
    TARGET_HEIGHT = 512

    resize_images_in_directory(INPUT_DIR, OUTPUT_DIR, TARGET_WIDTH, TARGET_HEIGHT)