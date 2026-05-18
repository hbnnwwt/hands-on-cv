"""第7章：Canny边缘检测与轮廓分析综合演示

教材对应：第7章 边缘检测与轮廓分析
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 添加父目录到路径以导入common模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_plot, show_images


def canny_demo(image: np.ndarray) -> None:
    """演示Canny边缘检测的不同参数"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges_low = cv2.Canny(gray, 50, 100)
    edges_mid = cv2.Canny(gray, 100, 200)
    edges_high = cv2.Canny(gray, 150, 300)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("原图")
    axes[1].imshow(edges_low, cmap='gray')
    axes[1].set_title("Canny(50,100) 灵敏")
    axes[2].imshow(edges_mid, cmap='gray')
    axes[2].set_title("Canny(100,200) 标准")
    axes[3].imshow(edges_high, cmap='gray')
    axes[3].set_title("Canny(150,300) 保守")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/canny_comparison.png", dpi=100)
    show_plot()


def sobel_demo(image: np.ndarray) -> None:
    """演示Sobel梯度计算"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # X方向、Y方向梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅值
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("原图")
    axes[1].imshow(np.abs(sobel_x), cmap='gray')
    axes[1].set_title("Sobel X (垂直边缘)")
    axes[2].imshow(np.abs(sobel_y), cmap='gray')
    axes[2].set_title("Sobel Y (水平边缘)")
    axes[3].imshow(magnitude, cmap='gray')
    axes[3].set_title("梯度幅值")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/sobel_demo.png", dpi=100)
    show_plot()


def contour_analysis(image: np.ndarray) -> None:
    """演示轮廓分析"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    print(f"检测到 {len(contours)} 个轮廓")

    # 计算每个轮廓的特征
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours[:10]):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)

        if area < 100:  # 跳过太小的
            continue

        # 圆度：4πS/L^2
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

        print(f"  轮廓{i}: 面积={area:.0f}, 周长={perimeter:.0f}, "
              f"边界框={w}x{h}, 圆度={circularity:.3f}")

        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/contours.png", vis)
    print("\n已保存轮廓可视化: outputs/contours.png")
    show_images(image, vis,
                titles=["原图", f"检测到 {len(contours)} 个轮廓"],
                window_size=(12, 5))


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "shapes.png")

    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取 {image_path}: {e}")
        sys.exit(1)

    Path("outputs").mkdir(exist_ok=True)

    print("=== Canny边缘检测对比 ===")
    canny_demo(image)
    print("\n=== Sobel梯度计算 ===")
    sobel_demo(image)
    print("\n=== 轮廓分析 ===")
    contour_analysis(image)


if __name__ == "__main__":
    main()
