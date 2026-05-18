"""第6章：透视变换矫正答题卡

教材对应：第6章 6.6 实战：答题卡透视矫正
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# 添加父目录到路径以导入common模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_images


def order_points(pts: np.ndarray) -> np.ndarray:
    """对4个点排序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)

    # 左上点 x+y 最小，右下点 x+y 最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上点 y-x 最小（x大y小），左下点 y-x 最大（x小y大）
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def find_paper_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """在图像中找出答题卡的四角"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 形态学闭运算填充小缺口
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 找最大轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    # 多边形近似到4个点
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) != 4:
        # 如果不是4个点，用最小外接矩形
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.int32)

    return approx.reshape(4, 2).astype(np.float32)


def perspective_correct(image: np.ndarray,
                        corners: np.ndarray,
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """根据角点做透视矫正

    参数：
        image: 原始图像
        corners: 4个角点（任意顺序）
        target_size: (width, height)，默认根据corners自动计算
    """
    # 角点排序
    rect = order_points(corners)
    tl, tr, br, bl = rect

    # 计算目标尺寸
    if target_size is None:
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_left, height_right))
    else:
        max_width, max_height = target_size

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


class SheetCorrector:
    """答题卡矫正器（封装为类，便于第17章集成）"""

    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        self.target_size = target_size

    def process(self, image: np.ndarray) -> Optional[np.ndarray]:
        """处理一张图像，返回矫正后的结果"""
        corners = find_paper_corners(image)
        if corners is None:
            print("[警告] 未检测到答题卡四角")
            return None

        return perspective_correct(image, corners, self.target_size)


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "tilted_paper.png")

    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取 {image_path}: {e}")
        sys.exit(1)

    print(f"处理图像: {image_path}, 尺寸: {image.shape}")

    # 单独跑一次 find_paper_corners 以便可视化
    corners = find_paper_corners(image)
    if corners is None:
        print("矫正失败：未检测到四角")
        sys.exit(1)

    corrected = perspective_correct(image, corners)
    print(f"矫正后尺寸: {corrected.shape}")

    Path("outputs").mkdir(exist_ok=True)
    from common.utils import save_image
    save_image(corrected, "outputs/corrected.png")
    print("已保存矫正图像: outputs/corrected.png")

    # 在原图上叠加检测到的四角
    overlay = image.copy()
    rect = order_points(corners).astype(np.int32)
    cv2.polylines(overlay, [rect], isClosed=True, color=(0, 255, 0), thickness=4)
    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip(rect, labels):
        cv2.circle(overlay, (int(x), int(y)), 12, (0, 0, 255), -1)
        cv2.putText(overlay, label, (int(x) + 15, int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    show_images(overlay, corrected,
                titles=["原图 + 检测到的四角", "透视矫正后"],
                window_size=(12, 6))


if __name__ == "__main__":
    main()
