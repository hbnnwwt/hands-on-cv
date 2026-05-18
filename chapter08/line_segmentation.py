"""第8章补充：文本行分割与多版式适配

教材对应：第8章 §8.3 / §8.7

内容：
  - 投影法文本行/列分割
  - 平滑与峰值检测
  - 模板匹配版式识别
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_plot, show_images


# ============================================================
# §8.3 文本行分割（投影法）
# ============================================================

def find_text_lines(horizontal: np.ndarray, threshold: int = 5,
                    min_height: int = 10) -> list:
    """从水平投影中找出文本行的位置

    参数：
        horizontal: 一维水平投影向量
        threshold: 判定为"有文字"的最小像素数
        min_height: 文本行的最小高度
    返回：
        [(y_start, y_end), ...]
    """
    in_text = False
    start = 0
    lines = []

    for y, value in enumerate(horizontal):
        if value > threshold and not in_text:
            in_text = True
            start = y
        elif value <= threshold and in_text:
            in_text = False
            if y - start >= min_height:
                lines.append((start, y))

    if in_text:
        lines.append((start, len(horizontal) - 1))

    return lines


def find_text_columns(vertical: np.ndarray, threshold: int = 5,
                      min_width: int = 5) -> list:
    """从垂直投影中找出文字列的位置（与 find_text_lines 对称）

    参数：
        vertical: 一维垂直投影向量
        threshold: 判定为"有内容"的最小像素数
        min_width: 列的最小宽度
    返回：
        [(x_start, x_end), ...]
    """
    in_text = False
    start = 0
    columns = []

    for x, value in enumerate(vertical):
        if value > threshold and not in_text:
            in_text = True
            start = x
        elif value <= threshold and in_text:
            in_text = False
            if x - start >= min_width:
                columns.append((start, x))

    if in_text:
        columns.append((start, len(vertical) - 1))

    return columns


def smooth_projection(projection: np.ndarray, window: int = 5
                      ) -> np.ndarray:
    """滑动平均平滑投影曲线"""
    kernel = np.ones(window) / window
    return np.convolve(projection, kernel, mode='same')


def segment_lines(binary_inv: np.ndarray,
                  threshold: int = 5,
                  min_height: int = 10,
                  window: int = 7) -> list:
    """从反转二值图中分割出所有文本行

    参数：
        binary_inv: 反转二值图（前景=255）
        threshold: 投影阈值
        min_height: 行最小高度
        window: 平滑窗口大小
    返回：
        [(y_start, y_end), ...]
    """
    horizontal = np.sum(binary_inv, axis=1) // 255
    if window > 1:
        horizontal = smooth_projection(horizontal, window)
    return find_text_lines(horizontal, threshold, min_height)


def segment_columns(binary_inv: np.ndarray,
                    threshold: int = 5,
                    min_width: int = 5,
                    window: int = 7) -> list:
    """从反转二值图中分割出所有文字列

    参数：
        binary_inv: 反转二值图（前景=255）
        threshold: 投影阈值
        min_width: 列最小宽度
        window: 平滑窗口大小
    返回：
        [(x_start, x_end), ...]
    """
    vertical = np.sum(binary_inv, axis=0) // 255
    if window > 1:
        vertical = smooth_projection(vertical, window)
    return find_text_columns(vertical, threshold, min_width)


# ============================================================
# §8.3 投影平滑 + 峰值检测（调试辅助）
# ============================================================

def smooth_and_find_peaks(projection: np.ndarray, window: int = 5,
                          min_distance: int = 15, height_ratio: float = 0.3
                          ) -> tuple:
    """平滑后做峰值检测，用于调试投影分割

    返回：
        (peaks_indices, smoothed_array)
    """
    kernel = np.ones(window) / window
    smoothed = np.convolve(projection, kernel, mode='same')

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(
        smoothed,
        height=smoothed.max() * height_ratio,
        distance=min_distance,
    )
    return peaks, smoothed


# ============================================================
# §8.7 模板匹配版式识别
# ============================================================

def identify_template(image: np.ndarray,
                      templates_dir: str,
                      resize: tuple = (200, 280)) -> tuple:
    """通过模板匹配识别答题卡版式

    参数：
        image: 待识别答题卡（已矫正）
        templates_dir: 存放各版式模板图的目录
        resize: 统一缩放尺寸
    返回：
        (best_name, best_score)
    """
    small = cv2.resize(image, resize)

    best_score = -1
    best_name = None

    for template_path in Path(templates_dir).glob("*.png"):
        try:
            template = read_image(str(template_path), grayscale=True)
        except (FileNotFoundError, ValueError):
            continue
        template = cv2.resize(template, resize)

        result = cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED)
        score = result.max()

        if score > best_score:
            best_score = score
            best_name = template_path.stem

    return best_name, best_score


# ============================================================
# 可视化辅助
# ============================================================

def draw_line_segments(image: np.ndarray, line_ranges: list,
                       color: tuple = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """在图像上绘制行/列分割线"""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, (start, end) in enumerate(line_ranges):
        # 判断是行分割（y坐标）还是列分割（x坐标）
        h, w = vis.shape[:2]
        if end <= h:  # 行坐标
            cv2.line(vis, (0, start), (w, start), color, thickness)
            cv2.line(vis, (0, end), (w, end), color, thickness)
            cv2.putText(vis, str(i), (5, start + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis


# ============================================================
# main
# ============================================================

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_1.png")

    try:
        image = read_image(image_path, grayscale=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取 {image_path}: {e}")
        sys.exit(1)

    Path("outputs").mkdir(exist_ok=True)

    # 二值化（反转：前景=255）
    _, binary_inv = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    print("=== 行分割 ===")
    line_ranges = segment_lines(binary_inv, threshold=5,
                                min_height=10, window=7)
    print(f"找到 {len(line_ranges)} 个文本行")
    for i, (y1, y2) in enumerate(line_ranges):
        print(f"  行{i + 1}: y={y1}~{y2}, 高度={y2 - y1}")

    vis_lines = draw_line_segments(image, line_ranges)
    cv2.imwrite("outputs/line_segmentation.png", vis_lines)
    print("已保存行分割结果: outputs/line_segmentation.png")

    print("\n=== 列分割 ===")
    col_ranges = segment_columns(binary_inv, threshold=5,
                                 min_width=5, window=7)
    print(f"找到 {len(col_ranges)} 个文字列")

    print("\n=== 峰值检测 ===")
    horizontal = np.sum(binary_inv, axis=1) // 255
    peaks, smoothed = smooth_and_find_peaks(horizontal)
    print(f"检测到 {len(peaks)} 个峰值位置: {peaks.tolist()}")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(horizontal, alpha=0.3, label='原始投影')
    ax.plot(smoothed, label='平滑投影')
    ax.plot(peaks, smoothed[peaks], 'rx', markersize=10, label='峰值')
    ax.set_title("水平投影 — 峰值检测")
    ax.set_xlabel("行索引")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/projection_peaks.png", dpi=100)
    show_plot()

    # 弹窗对比：原图 vs 行分割结果
    show_images(image, vis_lines,
                titles=["原图（灰度）", f"行分割结果（{len(line_ranges)} 行）"],
                window_size=(12, 6))


if __name__ == "__main__":
    main()
