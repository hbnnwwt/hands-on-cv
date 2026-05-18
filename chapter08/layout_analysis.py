"""第8章：版面分析与区域定位 — 综合演示

教材对应：第8章 版面分析与区域定位

核心模块：
  - 连通域分析（connected components）
  - 投影分析法（水平/垂直投影、文本行分割）
  - 霍夫直线检测与线段分类/聚类
  - 答题卡网格划分与版面分析 Pipeline
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_plot, show_images


# ============================================================
# §8.2 连通域分析
# ============================================================

def connected_components_demo(binary: np.ndarray) -> None:
    """演示连通域检测与筛选"""
    # 连通域标记（8-连通）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    print(f"找到 {num_labels - 1} 个连通域（不含背景）")

    # stats 的5列：[x, y, width, height, area]
    for i in range(1, min(num_labels, 11)):  # 只打印前10个
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        print(f"  组件{i}: 位置({x},{y}), 大小{w}x{h}, "
              f"面积{area}, 中心({cx:.1f},{cy:.1f})")

    # 筛选
    valid = filter_components(stats, centroids)
    print(f"\n筛选后剩余 {len(valid)} 个连通域")

    # 可视化
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for comp in valid:
        x, y, w, h = comp['bbox']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def filter_components(stats: np.ndarray, centroids: np.ndarray,
                      min_area: int = 50, max_area: int = 5000,
                      min_aspect: float = 0.3, max_aspect: float = 3.0
                      ) -> List[dict]:
    """筛选符合条件的连通域

    参数：
        stats: connectedComponentsWithStats 返回的统计信息
        min_area, max_area: 面积阈值
        min_aspect, max_aspect: 宽高比阈值（用于过滤细长线段）
    """
    valid = []
    for i in range(1, len(stats)):  # 跳过背景
        x, y, w, h, area = stats[i]

        if area < min_area or area > max_area:
            continue

        aspect = w / h if h > 0 else 0
        if aspect < min_aspect or aspect > max_aspect:
            continue

        valid.append({
            'id': i,
            'bbox': (x, y, w, h),
            'area': int(area),
            'center': tuple(centroids[i]),
        })
    return valid


# ============================================================
# §8.3 投影分析法
# ============================================================

def compute_projections(binary_inv: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """计算水平投影和垂直投影

    参数：
        binary_inv: 反转后的二值图（前景=255，背景=0）
    返回：
        (horizontal, vertical) 两个一维投影向量（像素计数）
    """
    horizontal = np.sum(binary_inv, axis=1) // 255  # shape: (H,)
    vertical = np.sum(binary_inv, axis=0) // 255    # shape: (W,)
    return horizontal, vertical


def smooth_projection(projection: np.ndarray, window: int = 5
                      ) -> np.ndarray:
    """对投影做滑动平均平滑"""
    kernel = np.ones(window) / window
    return np.convolve(projection, kernel, mode='same')


def find_text_lines(horizontal_projection: np.ndarray,
                    threshold: int = 5,
                    min_height: int = 10
                    ) -> List[Tuple[int, int]]:
    """从水平投影中找出文本行的位置

    参数：
        horizontal_projection: 一维水平投影向量
        threshold: 判定为"有文字"的最小像素数
        min_height: 文本行的最小高度
    返回：
        [(y_start, y_end), ...] 文本行的起始/结束行号
    """
    in_text = False
    start = 0
    lines = []

    for y, value in enumerate(horizontal_projection):
        if value > threshold and not in_text:
            in_text = True
            start = y
        elif value <= threshold and in_text:
            in_text = False
            if y - start >= min_height:
                lines.append((start, y))

    if in_text:
        lines.append((start, len(horizontal_projection) - 1))

    return lines


def projection_demo(binary: np.ndarray) -> None:
    """演示投影分析与文本行分割"""
    # 反转：前景（文字/标记）为白色255
    binary_inv = cv2.bitwise_not(binary)

    horizontal, vertical = compute_projections(binary_inv)
    smoothed = smooth_projection(horizontal, window=7)

    # 找文本行
    text_lines = find_text_lines(smoothed, threshold=5, min_height=10)
    print(f"找到 {len(text_lines)} 个文本行")
    for i, (y1, y2) in enumerate(text_lines):
        print(f"  行{i + 1}: y={y1}~{y2}, 高度={y2 - y1}")

    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(horizontal, alpha=0.4, label='原始')
    axes[0].plot(smoothed, label='平滑')
    axes[0].set_title("水平投影")
    axes[0].set_xlabel("行索引")
    axes[0].set_ylabel("白色像素数")
    axes[0].legend()

    axes[1].plot(vertical)
    axes[1].set_title("垂直投影")
    axes[1].set_xlabel("列索引")
    plt.tight_layout()
    plt.savefig("outputs/projection_demo.png", dpi=100)
    show_plot()


# ============================================================
# §8.4 霍夫变换检测直线 & 线段分类/聚类
# ============================================================

def classify_lines(lines: np.ndarray | None,
                   angle_tolerance: float = 10
                   ) -> Tuple[list, list, list]:
    """将检测到的线段按方向分类

    返回：horizontal, vertical, others
    """
    horizontal, vertical, others = [], [], []
    if lines is None:
        return horizontal, vertical, others

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angle < angle_tolerance or angle > 180 - angle_tolerance:
            horizontal.append(line[0])
        elif abs(angle - 90) < angle_tolerance:
            vertical.append(line[0])
        else:
            others.append(line[0])

    return horizontal, vertical, others


def cluster_lines(lines: list, axis: str = 'y',
                  merge_distance: float = 15) -> list:
    """将相近的平行线合并

    参数：
        lines: 线段列表 [(x1,y1,x2,y2), ...]
        axis: 'y' 表示按水平线的y坐标聚类
        merge_distance: 距离阈值，小于此距离的线被认为是同一条
    返回：
        合并后的坐标值列表（排序）
    """
    if not lines:
        return []

    if axis == 'y':
        coords = [(y1 + y2) / 2 for x1, y1, x2, y2 in lines]
    else:
        coords = [(x1 + x2) / 2 for x1, y1, x2, y2 in lines]

    coords.sort()

    clusters = [[coords[0]]]
    for c in coords[1:]:
        if c - clusters[-1][-1] < merge_distance:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    return [np.mean(cluster) for cluster in clusters]


def hough_lines_demo(binary: np.ndarray) -> np.ndarray:
    """演示霍夫直线检测与线段分类"""
    edges = cv2.Canny(binary, 50, 150)

    # 概率霍夫变换：返回线段端点 (x1,y1,x2,y2)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=80,
        minLineLength=binary.shape[1] // 4,
        maxLineGap=15,
    )

    # 分类
    horizontals, verticals, others = classify_lines(lines_p)
    h_grid = cluster_lines(horizontals, axis='y', merge_distance=20)
    v_grid = cluster_lines(verticals, axis='x', merge_distance=20)
    print(f"检测到水平线 {len(horizontals)} 条 → 聚类为 {len(h_grid)} 条")
    print(f"检测到垂直线 {len(verticals)} 条 → 聚类为 {len(v_grid)} 条")

    # 可视化
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("outputs/hough_lines.png", vis)
    print("已保存直线检测结果: outputs/hough_lines.png")
    return vis


# ============================================================
# §8.5 投影法版面区域检测
# ============================================================

def detect_regions_by_projection(binary: np.ndarray,
                                  min_gap_width: int = 35,
                                  smooth_window: int = 40,
                                  gap_threshold_ratio: float = 0.008
                                  ) -> List[dict]:
    """通过水平投影空隙自动检测版面区域

    原理：对二值图做水平投影，大窗口平滑后找空隙（投影值极低的行），
    空隙之间的连续内容块就是一个版面区域。

    参数：
        binary: 二值图（前景=255，背景=0，即 THRESH_BINARY_INV）
        min_gap_width: 空隙最小宽度（像素），小于此值不视为分界
        smooth_window: 投影平滑窗口大小
        gap_threshold_ratio: 空隙判定阈值（相对于图像宽度的比例）
    返回：
        [{"label": "info", "y_start": int, "y_end": int, "mean_density": float}, ...]
    """
    h, w = binary.shape
    h_proj = np.sum(binary, axis=1) // 255

    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(h_proj, kernel, mode='same')

    threshold = w * gap_threshold_ratio

    # 找空隙
    gaps = []
    in_gap = False
    gap_start = 0
    for y in range(h):
        if smoothed[y] < threshold and not in_gap:
            in_gap = True
            gap_start = y
        elif smoothed[y] >= threshold and in_gap:
            in_gap = False
            width = y - gap_start
            if width >= min_gap_width:
                gaps.append((gap_start, y))
    if in_gap and h - gap_start >= min_gap_width:
        gaps.append((gap_start, h))

    # 空隙之间就是区域
    boundaries = [0]
    for gs, ge in gaps:
        boundaries.extend([gs, ge])
    boundaries.append(h)

    regions = []
    for i in range(0, len(boundaries) - 1, 2):
        y_start = boundaries[i]
        y_end = boundaries[i + 1] if i + 1 < len(boundaries) else h
        height = y_end - y_start
        if height < 30:
            continue
        mean_density = float(np.mean(smoothed[y_start:y_end]))
        regions.append({
            "label": f"region_{len(regions)}",
            "y_start": y_start,
            "y_end": y_end,
            "height": height,
            "mean_density": mean_density,
        })

    return regions


def classify_regions(regions: List[dict],
                     image_height: int,
                     page: int = 1) -> List[dict]:
    """根据位置和特征对检测到的区域分类

    策略：
      - 过滤掉太小的区域（高度 < 图像高度的 3%）
      - Page1: 第一个=信息区, 最大的=选择题区, 最后一个=学号填涂区
      - Page2: 前半=判断题区, 后半=简答题区

    参数：
        regions: detect_regions_by_projection 的输出
        image_height: 图像高度
        page: 1 或 2
    返回：
        带有 "type" 和 "label" 字段的区域列表
    """
    if not regions:
        return regions

    # 过滤掉太小的区域（高度 < 图像高度的 3%）
    filtered = [r for r in regions if r["height"] > image_height * 0.03]

    if not filtered:
        return regions

    if page == 1:
        if len(filtered) >= 3:
            for i, r in enumerate(filtered):
                if i == 0:
                    r["type"] = "info"
                    r["label"] = "信息区"
                elif i == len(filtered) - 1 and r["y_start"] > image_height * 0.6:
                    r["type"] = "student_id"
                    r["label"] = "学号填涂区"
                else:
                    r["type"] = "choice"
                    r["label"] = "选择题区"
            filtered = _merge_same_type(filtered)
        elif len(filtered) == 2:
            filtered[0]["type"] = "info"
            filtered[0]["label"] = "信息区"
            filtered[1]["type"] = "choice"
            filtered[1]["label"] = "选择题区"
        else:
            filtered[0]["type"] = "choice"
            filtered[0]["label"] = "选择题区"
    else:
        if len(filtered) >= 2:
            mid = image_height * 0.45
            for r in filtered:
                if r["y_start"] < mid:
                    r["type"] = "judge"
                    r["label"] = "判断题区"
                else:
                    r["type"] = "essay"
                    r["label"] = "简答题区"
            # 合并同类型的连续区域
            merged = _merge_same_type(filtered)
            filtered = merged
        else:
            filtered[0]["type"] = "judge"
            filtered[0]["label"] = "判断题区"

    return filtered


def _merge_same_type(regions: List[dict]) -> List[dict]:
    """合并相邻的同类型区域"""
    if len(regions) <= 1:
        return regions
    merged = [regions[0].copy()]
    for r in regions[1:]:
        if r["type"] == merged[-1]["type"]:
            merged[-1]["y_end"] = r["y_end"]
            merged[-1]["height"] = r["y_end"] - merged[-1]["y_start"]
            merged[-1]["mean_density"] = (
                merged[-1]["mean_density"] + r["mean_density"]) / 2
        else:
            merged.append(r.copy())
    return merged


def draw_regions(image: np.ndarray, regions: List[dict]) -> np.ndarray:
    """在图像上绘制检测到的区域框和标签"""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    colors = {
        "info": (255, 178, 102),
        "choice": (102, 178, 255),
        "student_id": (178, 102, 255),
        "judge": (102, 255, 178),
        "essay": (255, 102, 178),
    }

    h, w = vis.shape[:2]
    for r in regions:
        y1, y2 = r["y_start"], r["y_end"]
        color = colors.get(r.get("type", ""), (0, 255, 0))
        cv2.rectangle(vis, (10, y1), (w - 10, y2), color, 3)
        label = f"{r['label']} (h={r['height']})"
        cv2.putText(vis, label, (20, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return vis


# ============================================================
# §8.6 答题卡网格划分
# ============================================================

# 标准答题卡尺寸（A4 300dpi）
STANDARD_WIDTH = 2480
STANDARD_HEIGHT = 3508

# 第一页（选择题+学号）的区域坐标
PAGE1_CONFIG = {
    "info_area":   {"x": 200,  "y": 200,  "w": 2080, "h": 300},
    "choice_area": {"x": 200,  "y": 600,  "w": 2080, "h": 1200},
}

# 第二页（判断题+简答题）的区域坐标
PAGE2_CONFIG = {
    "judge_area":  {"x": 200,  "y": 200,  "w": 2080, "h": 600},
    "essay_area":  {"x": 200,  "y": 900,  "w": 2080, "h": 1000},
}

# 兼容旧代码
LAYOUT_CONFIG = {**PAGE1_CONFIG, **PAGE2_CONFIG}


def generate_choice_grid(area: dict, num_questions: int = 20,
                         options_per_row: int = 4,
                         questions_per_row: int = 4,
                         bubble_size: int = 40) -> List[dict]:
    """在选择题区域生成所有气泡的坐标

    参数：
        area: 选择题区域 {x, y, w, h}
        num_questions: 题目数量
        options_per_row: 每题选项数（如A/B/C/D=4）
        questions_per_row: 每行题数
        bubble_size: 气泡直径
    返回：
        [{"question_id": i, "option": "A", "bbox": (x,y,w,h)}, ...]
    """
    bubbles = []
    num_rows = (num_questions + questions_per_row - 1) // questions_per_row

    h_per_question = area["w"] / questions_per_row
    v_per_row = area["h"] / num_rows

    for q in range(num_questions):
        row = q // questions_per_row
        col = q % questions_per_row

        q_x = area["x"] + col * h_per_question
        q_y = area["y"] + row * v_per_row

        option_start_x = q_x + h_per_question / 3
        option_spacing = (h_per_question * 2 / 3) / options_per_row

        for opt_idx, opt_letter in enumerate(
                ['A', 'B', 'C', 'D'][:options_per_row]):
            bubble_cx = option_start_x + opt_idx * option_spacing
            bubble_cy = q_y + v_per_row / 2

            bubbles.append({
                "question_id": q + 1,
                "option": opt_letter,
                "bbox": (
                    int(bubble_cx - bubble_size / 2),
                    int(bubble_cy - bubble_size / 2),
                    bubble_size,
                    bubble_size,
                )
            })

    return bubbles


def generate_judge_grid(area: dict, num_questions: int = 10,
                        questions_per_row: int = 4,
                        question_start: int = 21,
                        bubble_size: int = 40) -> List[dict]:
    """在判断题区域生成所有气泡的坐标（T/F选项）

    参数：
        area: 判断题区域 {x, y, w, h}
        num_questions: 判断题数量（默认10）
        questions_per_row: 每行题数
        question_start: 起始题号（默认21，接在选择题之后）
        bubble_size: 气泡直径
    返回：
        [{"question_id": i, "option": "T", "bbox": (x,y,w,h)}, ...]
    """
    bubbles = []
    options = ['T', 'F']
    num_rows = (num_questions + questions_per_row - 1) // questions_per_row

    h_per_question = area["w"] / questions_per_row
    v_per_row = area["h"] / num_rows

    for q in range(num_questions):
        row = q // questions_per_row
        col = q % questions_per_row

        q_x = area["x"] + col * h_per_question
        q_y = area["y"] + row * v_per_row

        # T/F 只有2个选项，放在题号右侧
        option_start_x = q_x + h_per_question / 3
        option_spacing = (h_per_question * 2 / 3) / len(options)

        for opt_idx, opt_label in enumerate(options):
            bubble_cx = option_start_x + opt_idx * option_spacing
            bubble_cy = q_y + v_per_row / 2

            bubbles.append({
                "question_id": question_start + q,
                "option": opt_label,
                "bbox": (
                    int(bubble_cx - bubble_size / 2),
                    int(bubble_cy - bubble_size / 2),
                    bubble_size,
                    bubble_size,
                )
            })

    return bubbles


def extract_bubble_rois(image: np.ndarray, bubbles: List[dict]
                        ) -> List[dict]:
    """从图像中提取所有气泡的 ROI

    返回：
        [{...bubble, "roi": np.ndarray}, ...]
    """
    results = []
    for bubble in bubbles:
        x, y, w, h = bubble["bbox"]
        # 边界检查
        ih, iw = image.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)
        roi = image[y:y + h, x:x + w]
        results.append({**bubble, "roi": roi})
    return results


# ============================================================
# §8.6 版面分析完整 Pipeline
# ============================================================

@dataclass
class Region:
    """版面区域"""
    type: str       # "choice_bubble", "judge_box", "essay_line"
    bbox: tuple     # (x, y, w, h)
    metadata: dict  # 如 {"question": 1, "option": "A"}


class LayoutAnalyzer:
    """答题卡版面分析器（支持两页试卷）"""

    def __init__(self, config: dict | None = None):
        self.config = config or LAYOUT_CONFIG

    def analyze_page1(self, corrected_image: np.ndarray) -> List[Region]:
        """分析第一页：学号区域 + 选择题区域"""
        regions = []

        # 选择题区域 → 生成网格
        choice_cfg = self.config['choice_area']
        bubbles = generate_choice_grid(choice_cfg)
        for b in bubbles:
            regions.append(Region(
                type="choice_bubble",
                bbox=b["bbox"],
                metadata={"question": b["question_id"],
                          "option": b["option"]}
            ))

        return regions

    def analyze_page2(self, corrected_image: np.ndarray) -> List[Region]:
        """分析第二页：判断题区域 + 简答题区域"""
        regions = []

        # 判断题区域 → 生成T/F网格
        judge_cfg = self.config['judge_area']
        judge_bubbles = generate_judge_grid(judge_cfg)
        for b in judge_bubbles:
            regions.append(Region(
                type="judge_bubble",
                bbox=b["bbox"],
                metadata={"question": b["question_id"],
                          "option": b["option"]}
            ))

        # 简答题区域 → 投影分割文本行
        essay_cfg = self.config['essay_area']
        ex, ey = essay_cfg['x'], essay_cfg['y']
        ew, eh = essay_cfg['w'], essay_cfg['h']
        essay_img = corrected_image[ey:ey + eh, ex:ex + ew]
        line_ranges = locate_essay_lines(essay_img)
        for i, (y1, y2) in enumerate(line_ranges):
            regions.append(Region(
                type="essay_line",
                bbox=(ex, ey + y1, ew, y2 - y1),
                metadata={"line_index": i}
            ))

        return regions

    def analyze(self, corrected_image: np.ndarray) -> List[Region]:
        """兼容旧接口：分析第一页"""
        return self.analyze_page1(corrected_image)


def locate_essay_lines(essay_area_image: np.ndarray,
                       threshold: int = 5,
                       min_height: int = 20
                       ) -> List[Tuple[int, int]]:
    """在简答题区域内定位每一行文字

    返回：
        [(y1, y2), ...] 每行文字的纵坐标范围
    """
    gray = (cv2.cvtColor(essay_area_image, cv2.COLOR_BGR2GRAY)
            if essay_area_image.ndim == 3 else essay_area_image)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    horizontal = np.sum(binary, axis=1) // 255
    lines = find_text_lines(horizontal, threshold, min_height)
    return lines


def save_layout(layout: List[Region], filepath: str) -> None:
    """保存版面分析结果为 JSON"""
    serializable = []
    for region in layout:
        serializable.append({
            "type": region.type,
            "bbox": list(region.bbox),
            "metadata": region.metadata,
        })
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# ============================================================
# 调试辅助
# ============================================================

def smooth_and_find_peaks(projection: np.ndarray, window: int = 5,
                          min_distance: int = 15
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """先平滑，再做峰值检测"""
    kernel = np.ones(window) / window
    smoothed = np.convolve(projection, kernel, mode='same')

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(
        smoothed,
        height=smoothed.max() * 0.3,
        distance=min_distance,
    )
    return peaks, smoothed


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

    # 旋转校正：确保版面分析输入是正向的
    original = image.copy()
    from chapter07.canny_and_contours import correct_rotation
    corrected = correct_rotation(image)
    if corrected is not None:
        image = corrected

    # 二值化
    _, binary = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    print("=== §8.2 连通域分析 ===")
    vis = connected_components_demo(binary)
    cv2.imwrite("outputs/connected_components.png", vis)
    print("已保存连通域可视化: outputs/connected_components.png")

    print("\n=== §8.3 投影分析法 ===")
    projection_demo(binary)

    print("\n=== §8.4 霍夫直线检测 ===")
    hough_vis = hough_lines_demo(image)

    # 可视化连通域 + 霍夫直线对比
    show_images(original, image, hough_vis,
                titles=["校正前", "校正后", "霍夫直线"],
                window_size=(15, 5))

    print("\n=== §8.5 投影法版面区域检测 ===")
    # 第一页：自动检测区域
    raw_regions_p1 = detect_regions_by_projection(binary)
    regions_p1 = classify_regions(raw_regions_p1, image.shape[0], page=1)
    print(f"第一页检测到 {len(regions_p1)} 个区域：")
    for r in regions_p1:
        print(f"  {r['label']}: y={r['y_start']}~{r['y_end']}, "
              f"高度={r['height']}, 密度={r['mean_density']:.0f}")

    vis_regions_p1 = draw_regions(image, regions_p1)
    cv2.imwrite("outputs/layout_regions_p1.png", vis_regions_p1)
    print("已保存: outputs/layout_regions_p1.png")

    # 第二页
    repo_root = Path(__file__).parent.parent
    page2_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_2.png")
    try:
        page2_image = read_image(page2_path, grayscale=True)
    except (FileNotFoundError, ValueError):
        page2_image = None

    if page2_image is not None:
        page2_corrected = correct_rotation(page2_image)
        if page2_corrected is not None:
            page2_image = page2_corrected
        _, binary_p2 = cv2.threshold(page2_image, 0, 255,
                                     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        raw_regions_p2 = detect_regions_by_projection(binary_p2)
        regions_p2 = classify_regions(raw_regions_p2, page2_image.shape[0], page=2)
        print(f"\n第二页检测到 {len(regions_p2)} 个区域：")
        for r in regions_p2:
            print(f"  {r['label']}: y={r['y_start']}~{r['y_end']}, "
                  f"高度={r['height']}, 密度={r['mean_density']:.0f}")

        vis_regions_p2 = draw_regions(page2_image, regions_p2)
        cv2.imwrite("outputs/layout_regions_p2.png", vis_regions_p2)
        print("已保存: outputs/layout_regions_p2.png")

        show_images(image, vis_regions_p1, page2_image, vis_regions_p2,
                    titles=["第一页", "第一页区域", "第二页", "第二页区域"],
                    window_size=(16, 8))
    else:
        print("未找到第二页模板，跳过第二页分析")
        show_images(image, vis_regions_p1,
                    titles=["第一页", "第一页区域"],
                    window_size=(12, 6))

    print("\n=== §8.6 版面分析 Pipeline ===")
    analyzer = LayoutAnalyzer()
    regions_p1_bubbles = analyzer.analyze_page1(image)
    print(f"第一页生成 {len(regions_p1_bubbles)} 个气泡区域")
    save_layout(regions_p1_bubbles, "outputs/layout_page1.json")
    print("已保存: outputs/layout_page1.json")


if __name__ == "__main__":
    main()
