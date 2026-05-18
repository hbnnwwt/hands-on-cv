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


def correct_rotation(image: np.ndarray) -> np.ndarray | None:
    """通过轮廓方向角检测并校正图像旋转

    原理：找到图像中最大轮廓，用 minAreaRect 获取其旋转角度，
    再用仿射变换将图像旋转到正向。适用于答题卡等有明显外轮廓的文档。

    假设纸张已处于正确朝向（竖版：h > w），仅修正小角度倾斜。
    将 minAreaRect 的角度归一化到 [-45°, 45°) 后取反校正。

    参数：
        image: 输入图像（灰度或彩色）

    返回：
        校正后的图像，检测失败返回 None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < gray.shape[0] * gray.shape[1] * 0.1:
        return None

    rect = cv2.minAreaRect(largest)
    (_, (w, h), angle) = rect

    # 归一化角度到 [-45°, 45°)
    # minAreaRect 可能返回 (w>h, angle≈0) 或 (w<h, angle≈-90)
    # 两者描述的是同一个纸张，归一化后只保留倾斜分量
    while angle > 45:
        angle -= 90
    while angle <= -45:
        angle += 90

    rotation = angle

    if abs(rotation) < 0.5:
        return image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    (ih, iw) = image.shape[:2]
    center = (iw // 2, ih // 2)
    M = cv2.getRotationMatrix2D(center, rotation, 1.0)

    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int(ih * sin_val + iw * cos_val)
    new_h = int(ih * cos_val + iw * sin_val)

    M[0, 2] += (new_w - iw) / 2
    M[1, 2] += (new_h - ih) / 2

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    print(f"旋转校正: {rotation:.1f}°, "
          f"尺寸 {iw}x{ih} -> {new_w}x{new_h}")
    return rotated


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
        image_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_1.png")

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

    # 旋转校正演示
    repo_root = Path(__file__).parent.parent
    tilted_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_1.png")
    try:
        tilted = read_image(tilted_path)
        print("\n=== 旋转校正 ===")
        corrected = correct_rotation(tilted)
        if corrected is not None:
            from common.utils import save_image
            save_image(corrected, "outputs/rotation_corrected.png")
            show_images(tilted, corrected,
                        titles=["校正前", "旋转校正后"],
                        window_size=(12, 6))
    except (FileNotFoundError, ValueError):
        print(f"\n跳过旋转校正演示（未找到 {tilted_path}）")


if __name__ == "__main__":
    main()
