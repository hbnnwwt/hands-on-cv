"""第3章：NumPy图像数组操作综合演示

教材对应：第3章 Python科学计算基础
"""

from __future__ import annotations
import time
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 添加父目录到路径以导入common模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_images


def perf_compare():
    """演示NumPy向量化 vs Python循环的性能差距"""
    print("=== 性能对比：循环 vs 向量化 ===")
    size = 1_000_000

    # Python列表
    py_list = list(range(size))
    start = time.time()
    _ = [x + 10 for x in py_list]
    time_list = time.time() - start

    # NumPy
    np_array = np.arange(size)
    start = time.time()
    _ = np_array + 10
    time_numpy = time.time() - start

    print(f"Python列表: {time_list * 1000:.2f} ms")
    print(f"NumPy数组:  {time_numpy * 1000:.2f} ms")
    print(f"加速比: {time_list / max(time_numpy, 1e-9):.0f}x")


def numpy_image_ops(image_path: str):
    """演示NumPy操作图像数组"""
    try:
        img = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"无法读取 {image_path}: {e}")
        return

    print(f"\n=== NumPy图像操作 ===")
    print(f"shape: {img.shape}")
    print(f"dtype: {img.dtype}")
    print(f"size: {img.size}")
    print(f"通道数: {img.ndim}")

    # 切片 = 裁剪
    h, w = img.shape[:2]
    cropped = img[h // 4:3 * h // 4, w // 4:3 * w // 4]
    print(f"\n裁剪后: {cropped.shape}")

    # 算术 = 亮度调节
    brighter = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    darker = np.clip(img.astype(np.int16) - 50, 0, 255).astype(np.uint8)

    # 布尔索引 = 二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = np.zeros_like(gray)
    binary[gray > 127] = 255

    # 广播 = 通道加权
    weights = np.array([0.114, 0.587, 0.299])   # BGR顺序
    gray_manual = np.sum(img.astype(np.float32) * weights, axis=2)
    gray_manual = gray_manual.astype(np.uint8)

    # 保存
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/np_brighter.png", brighter)
    cv2.imwrite("outputs/np_binary.png", binary)
    cv2.imwrite("outputs/np_gray_manual.png", gray_manual)
    print("\n已保存:")
    print("  - outputs/np_brighter.png (NumPy加法调亮)")
    print("  - outputs/np_binary.png  (布尔索引二值化)")
    print("  - outputs/np_gray_manual.png (广播灰度转换)")

    # 可视化对比
    show_images(img, brighter, darker, binary, gray_manual,
                titles=["原图", "+50 调亮", "-50 调暗", "布尔二值化", "广播灰度"],
                window_size=(15, 4))


def broadcasting_demo():
    """演示广播机制"""
    print("\n=== 广播机制 ===")

    # 标量 + 数组
    a = np.array([1, 2, 3])
    print(f"{a} + 10 = {a + 10}")

    # 一维 + 二维
    matrix = np.ones((3, 3))
    row = np.array([1, 2, 3])
    print(f"\nmatrix:\n{matrix}")
    print(f"row: {row}")
    print(f"matrix * row:\n{matrix * row}")


def stats_demo():
    """演示统计函数"""
    print("\n=== 统计函数 ===")
    repo_root = Path(__file__).parent.parent
    image_path = repo_root / "data" / "imgs" / "test_img.bmp"

    if not image_path.exists():
        print("跳过：测试图像不存在")
        return

    try:
        gray = read_image(str(image_path), grayscale=True)
        print(f"均值: {gray.mean():.2f}")
        print(f"标准差: {gray.std():.2f}")
        print(f"最小值: {gray.min()}")
        print(f"最大值: {gray.max()}")

        # 各通道均值
        color = read_image(str(image_path))
        bgr_mean = color.mean(axis=(0, 1))
        print(f"BGR各通道均值: B={bgr_mean[0]:.1f}, G={bgr_mean[1]:.1f}, R={bgr_mean[2]:.1f}")
    except (FileNotFoundError, ValueError) as e:
        print(f"跳过：无法读取图像 - {e}")


def main():
    perf_compare()
    broadcasting_demo()
    stats_demo()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "imgs" / "test_img.bmp")
        if not Path(image_path).exists():
            print("\n提示：运行 python common/test_images.py 生成测试图像")
            return

    numpy_image_ops(image_path)


if __name__ == "__main__":
    main()
