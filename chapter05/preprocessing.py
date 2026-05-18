"""第5章：图像预处理 — 综合演示

教材对应：第5章 图像预处理与增强
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 添加父目录到路径以导入common模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_plot


def add_noise(image: np.ndarray, noise_type: str = 'gaussian',
              **kwargs) -> np.ndarray:
    """向图像添加噪声"""
    if noise_type == 'gaussian':
        sigma = kwargs.get('sigma', 25)
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
    elif noise_type == 'salt_pepper':
        prob = kwargs.get('prob', 0.05)
        noisy = image.copy()
        # Salt
        salt = np.random.random(image.shape[:2]) < prob / 2
        noisy[salt] = 255
        # Pepper
        pepper = np.random.random(image.shape[:2]) < prob / 2
        noisy[pepper] = 0
    else:
        raise ValueError(f"未知噪声类型: {noise_type}")
    return np.clip(noisy, 0, 255).astype(np.uint8)


def compare_filters(image: np.ndarray) -> None:
    """对比四种滤波器的效果"""
    # 添加噪声
    noisy = add_noise(image, 'salt_pepper', prob=0.05)

    # 四种滤波
    mean_filter = cv2.blur(noisy, (5, 5))
    gaussian = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)
    median = cv2.medianBlur(noisy, 5)
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    titles_imgs = [
        ("原图", image),
        ("噪声图（椒盐）", noisy),
        ("均值滤波", mean_filter),
        ("高斯滤波", gaussian),
        ("中值滤波（最佳）", median),
        ("双边滤波", bilateral),
    ]

    for ax, (title, img) in zip(axes.flat, titles_imgs):
        ax.imshow(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/filter_comparison.png", dpi=100)
    show_plot()


def threshold_demo(image: np.ndarray) -> None:
    """对比不同二值化方法"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 全局固定阈值
    _, binary_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Otsu自动阈值
    _, binary_otsu = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 自适应高斯阈值
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("原灰度图")
    axes[1].imshow(binary_fixed, cmap='gray')
    axes[1].set_title("固定阈值=127")
    axes[2].imshow(binary_otsu, cmap='gray')
    axes[2].set_title("Otsu自动")
    axes[3].imshow(binary_adaptive, cmap='gray')
    axes[3].set_title("自适应")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/threshold_comparison.png", dpi=100)
    show_plot()


def morphology_demo(binary: np.ndarray) -> None:
    """演示形态学运算"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    dilation = cv2.dilate(binary, kernel)
    erosion = cv2.erode(binary, kernel)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles_imgs = [
        ("原图", binary),
        ("膨胀", dilation),
        ("腐蚀", erosion),
        ("开运算", opening),
        ("闭运算", closing),
        ("形态学梯度", gradient),
    ]
    for ax, (title, img) in zip(axes.flat, titles_imgs):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/morphology_demo.png", dpi=100)
    show_plot()


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_1.png")

    Path("outputs").mkdir(exist_ok=True)

    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取 {image_path}: {e}")
        return

    print("=== 滤波器对比 ===")
    compare_filters(image)
    print("\n=== 二值化对比 ===")
    threshold_demo(image)

    # 用于形态学的二值图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print("\n=== 形态学运算 ===")
    morphology_demo(binary)


if __name__ == "__main__":
    main()
