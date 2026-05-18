"""第2章：图像读取、显示、保存

教材对应：第2章 2.4 OpenCV基础操作
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

# 设置路径以导入共享工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_images


def demo_read_show(image_path: str) -> None:
    """演示图像读取与显示"""
    # 方法1：彩色读取
    img_color = read_image(image_path)
    print(f"彩色图像 shape: {img_color.shape}, dtype: {img_color.dtype}")

    # 方法2：灰度读取
    img_gray = read_image(image_path, grayscale=True)
    print(f"灰度图像 shape: {img_gray.shape}, dtype: {img_gray.dtype}")

    # 方法3：保留Alpha通道（read_image 不支持 UNCHANGED，需要兜底）
    try:
        img_rgba = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        # 如果中文路径兜底失败，尝试标准方式
        img_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"原始读取 shape: {img_rgba.shape}")


def demo_pixel_access(image_path: str) -> None:
    """演示像素访问与修改"""
    img = read_image(image_path)
    h, w = img.shape[:2]

    # 访问单个像素
    cy, cx = h // 2, w // 2
    pixel = img[cy, cx]
    print(f"中心像素 (B={pixel[0]}, G={pixel[1]}, R={pixel[2]})")

    # 修改一片区域（ROI）
    img_modified = img.copy()
    img_modified[100:200, 100:200] = [0, 255, 0]  # 绿色方块
    return img_modified


def demo_crop(image_path: str, output_path: str = "cropped.png") -> np.ndarray:
    """演示图像裁剪（ROI操作）"""
    img = read_image(image_path)
    h, w = img.shape[:2]

    # 裁剪中心区域
    cropped = img[h // 4:3 * h // 4, w // 4:3 * w // 4]

    cv2.imwrite(output_path, cropped)
    print(f"已保存裁剪后图像到 {output_path}")
    print(f"原图: {img.shape} -> 裁剪后: {cropped.shape}")
    return cropped


def demo_drawing(image_path: str, output_path: str = "annotated.png") -> np.ndarray:
    """演示在图像上绘图"""
    img = read_image(image_path).copy()
    h, w = img.shape[:2]

    # 矩形
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)

    # 圆形
    cv2.circle(img, (w - 100, 100), 50, (255, 0, 0), -1)

    # 直线
    cv2.line(img, (0, h - 100), (w, h - 100), (0, 0, 255), 2)

    # 文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "OpenCV Demo", (50, h - 30), font, 1, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"已保存标注图像到 {output_path}")
    return img


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "imgs" / "test_img.bmp")

    print("=== 第2章 图像基础操作演示 ===\n")
    demo_read_show(image_path)
    print()
    modified = demo_pixel_access(image_path)
    print()
    cropped = demo_crop(image_path, "outputs/cropped.png")
    print()
    annotated = demo_drawing(image_path, "outputs/annotated.png")

    # 可视化对比
    original = read_image(image_path)
    show_images(original, modified, cropped, annotated,
                titles=["原图", "像素修改", "ROI裁剪", "绘图标注"],
                window_size=(14, 5))


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    main()
