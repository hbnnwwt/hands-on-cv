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
# §8.5 形态学闭运算 + 轮廓检测的版面区域定位
# ============================================================

def detect_regions_by_morphology(binary: np.ndarray,
                                  min_area_ratio: float = 0.03,
                                  max_area_ratio: float = 0.92,
                                  kernel_ratio: float = 0.02
                                  ) -> List[dict]:
    """通过形态学闭运算检测答题卡的带边框区域

    原理：答题卡的每个区域（学号、选择题等）都有矩形边框。
    反转二值图后做闭运算，将边框与内部内容融合成大块连通区域，
    再用轮廓检测定位。取面积最大的 2 个区域。

    参数：
        binary: 二值图（前景=255，背景=0，即 THRESH_BINARY_INV）
        min_area_ratio: 最小面积占比（过滤噪声小区域）
        max_area_ratio: 最大面积占比（过滤整页轮廓）
        kernel_ratio: 闭运算核大小占图像高度的比例
    返回：
        [{"label": str, "bbox": (x, y, w, h)}, ...] 按y坐标排序
    """
    h, w = binary.shape
    total_area = h * w

    # 反转：边框和内容变白，背景变黑
    inv = 255 - binary

    # 形态学闭运算：合并边框线与内部内容为一个连通区域
    ksize = max(int(h * kernel_ratio), 5)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)

    # 轮廓检测
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect < 0.3 or aspect > 5.0:
            continue
        boxes.append((x, y, bw, bh))

    # 按面积降序取最大的2个，再按y坐标排序
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:2]
    boxes.sort(key=lambda b: b[1])

    regions = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        regions.append({
            "label": f"region_{i}",
            "bbox": (x, y, bw, bh),
            "area": bw * bh,
        })

    return regions, closed


def fallback_regions(h: int, w: int, page: int = 1) -> List[dict]:
    """自动检测失败时使用固定比例的 fallback 区域

    参考实际答题卡尺寸标定：Page1 学号区占 6%~26%，选择题区占 28%~80%；
    Page2 判断题区占 6%~46%，简答题区占 50%~90%。

    参数：
        h: 图像高度
        w: 图像宽度
        page: 1 或 2
    """
    if page == 1:
        configs = [
            {"type": "student_id", "label": "学号区", "range": (0.06, 0.26)},
            {"type": "choice",     "label": "选择题区", "range": (0.28, 0.80)},
        ]
    else:
        configs = [
            {"type": "judge", "label": "判断题区", "range": (0.06, 0.46)},
            {"type": "essay", "label": "简答题区", "range": (0.50, 0.90)},
        ]

    regions = []
    for cfg in configs:
        y_start = int(h * cfg["range"][0])
        y_end = int(h * cfg["range"][1])
        regions.append({
            "type": cfg["type"],
            "label": cfg["label"],
            "bbox": (0, y_start, w, y_end - y_start),
            "area": w * (y_end - y_start),
        })

    return regions


def classify_regions_morph(regions: List[dict], page: int = 1) -> List[dict]:
    """为检测到的区域分配类型

    Page1: 上方=学号区，下方=选择题区
    Page2: 上方=判断题区，下方=简答题区
    """
    if not regions:
        return regions

    if page == 1:
        types = ["student_id", "choice"]
        labels = ["学号区", "选择题区"]
    else:
        types = ["judge", "essay"]
        labels = ["判断题区", "简答题区"]

    for i, r in enumerate(regions):
        if i < len(types):
            r["type"] = types[i]
            r["label"] = labels[i]
        else:
            r["type"] = "other"
            r["label"] = f"其他区域{i}"

    return regions


def draw_regions(image: np.ndarray, regions: List[dict]) -> np.ndarray:
    """在图像上绘制检测到的区域框和标签"""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    colors = {
        "student_id": (0, 200, 0),
        "choice": (255, 100, 0),
        "judge": (0, 150, 255),
        "essay": (200, 0, 200),
    }

    for r in regions:
        x, y, bw, bh = r["bbox"]
        color = colors.get(r.get("type", ""), (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(vis, r["label"], (x + 5, y + 30),
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

    print("\n=== §8.5 形态学闭运算 + 轮廓检测 ===")
    # 第一页：自动检测区域
    raw_regions_p1, morph_p1 = detect_regions_by_morphology(binary)
    if len(raw_regions_p1) >= 2:
        regions_p1 = classify_regions_morph(raw_regions_p1, page=1)
        print("第一页：轮廓检测成功")
    else:
        regions_p1 = fallback_regions(image.shape[0], image.shape[1], page=1)
        print("第一页：轮廓检测失败，使用固定比例 fallback")
    print(f"第一页检测到 {len(regions_p1)} 个区域：")
    for r in regions_p1:
        x, y, bw, bh = r["bbox"]
        print(f"  {r['label']}: ({x},{y}) {bw}x{bh}")

    vis_regions_p1 = draw_regions(image, regions_p1)
    cv2.imwrite("outputs/layout_regions_p1.png", vis_regions_p1)
    cv2.imwrite("outputs/morph_p1.png", morph_p1)
    print("已保存: outputs/layout_regions_p1.png, outputs/morph_p1.png")

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

        raw_regions_p2, morph_p2 = detect_regions_by_morphology(binary_p2)
        if len(raw_regions_p2) >= 2:
            regions_p2 = classify_regions_morph(raw_regions_p2, page=2)
            print("\n第二页：轮廓检测成功")
        else:
            regions_p2 = fallback_regions(page2_image.shape[0], page2_image.shape[1], page=2)
            print("\n第二页：轮廓检测失败，使用固定比例 fallback")
        print(f"\n第二页检测到 {len(regions_p2)} 个区域：")
        for r in regions_p2:
            x, y, bw, bh = r["bbox"]
            print(f"  {r['label']}: ({x},{y}) {bw}x{bh}")

        vis_regions_p2 = draw_regions(page2_image, regions_p2)
        cv2.imwrite("outputs/layout_regions_p2.png", vis_regions_p2)
        cv2.imwrite("outputs/morph_p2.png", morph_p2)
        print("已保存: outputs/layout_regions_p2.png, outputs/morph_p2.png")

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
