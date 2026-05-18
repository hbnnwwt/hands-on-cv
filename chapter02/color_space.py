"""第2章：色彩空间转换

教材对应：第2章 2.2 色彩的科学
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置路径以导入共享工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_plot


def show_channels(image_path: str) -> None:
    """显示彩色图像的各个通道"""
    img = read_image(image_path)
    b, g, r = cv2.split(img)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("原图")

    axes[0, 1].imshow(r, cmap='Reds')
    axes[0, 1].set_title("R通道")

    axes[1, 0].imshow(g, cmap='Greens')
    axes[1, 0].set_title("G通道")

    axes[1, 1].imshow(b, cmap='Blues')
    axes[1, 1].set_title("B通道")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    show_plot("outputs/channels.png")


def gray_conversion(image_path: str) -> None:
    """演示三种灰度转换方式"""
    img = read_image(image_path)

    # 方法1：OpenCV内置（推荐）
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 方法2：手动加权（ITU-R BT.601）
    weights = np.array([0.114, 0.587, 0.299])  # BGR顺序
    gray_manual = np.sum(img.astype(np.float32) * weights, axis=2)
    gray_manual = np.clip(gray_manual, 0, 255).astype(np.uint8)

    # 方法3：平均（不推荐，颜色感知不准确）
    gray_avg = np.mean(img, axis=2).astype(np.uint8)

    print(f"OpenCV灰度  最大差异（与手动）: {np.abs(gray_cv.astype(int) - gray_manual.astype(int)).max()}")
    print(f"平均法灰度  最大差异（与OpenCV）: {np.abs(gray_cv.astype(int) - gray_avg.astype(int)).max()}")

    cv2.imwrite("outputs/gray_opencv.png", gray_cv)
    cv2.imwrite("outputs/gray_manual.png", gray_manual)


def hsv_demo(image_path: str) -> None:
    """演示HSV色彩空间"""
    img = read_image(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原图")
    axes[1].imshow(h, cmap='hsv')
    axes[1].set_title("H通道（色相）")
    axes[2].imshow(s, cmap='gray')
    axes[2].set_title("S通道（饱和度）")
    axes[3].imshow(v, cmap='gray')
    axes[3].set_title("V通道（亮度）")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    show_plot("outputs/hsv_channels.png")


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "imgs" / "test_img.bmp")

    Path("outputs").mkdir(exist_ok=True)

    print("=== 通道分离 ===")
    show_channels(image_path)
    print("\n=== 灰度转换对比 ===")
    gray_conversion(image_path)
    print("\n=== HSV色彩空间 ===")
    hsv_demo(image_path)


if __name__ == "__main__":
    main()
