"""第15章：IoU、NMS、目标检测基础工具

教材对应：第15章 15.1 目标检测概述
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import show_plot, show_images


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float],
                eps: float = 1e-6) -> float:
    """计算两个边界框的IoU

    参数：
        box1, box2: (x1, y1, x2, y2) 格式
        eps: 防止除零的小常数

    返回：
        IoU 值 (0~1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + eps)


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = 0.5) -> List[int]:
    """非极大值抑制

    参数：
        boxes: shape (N, 4)，每行 (x1, y1, x2, y2)
        scores: shape (N,)，每个框的置信度
        iou_threshold: IoU 阈值

    返回：
        保留的索引列表
    """
    if len(boxes) == 0:
        return []

    # 按置信度降序
    order = np.argsort(scores)[::-1].tolist()
    keep = []

    while order:
        # 当前最高分的框
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # 计算与剩余所有框的IoU
        rest = order[1:]
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in rest])

        # 保留IoU < threshold的
        order = [rest[k] for k in range(len(rest)) if ious[k] < iou_threshold]

    return keep


def soft_nms(boxes: np.ndarray, scores: np.ndarray,
             sigma: float = 0.5,
             score_threshold: float = 0.001) -> Tuple[List[int], np.ndarray]:
    """Soft-NMS（高斯衰减版本）

    与硬NMS的区别：不直接删除高IoU框，而是降低其分数
    """
    if len(boxes) == 0:
        return [], np.array([])

    scores = scores.copy()
    keep = []
    indices = list(range(len(boxes)))

    while indices:
        # 当前最高分
        max_idx = max(indices, key=lambda i: scores[i])
        keep.append(max_idx)
        indices.remove(max_idx)

        # 对剩余框做高斯衰减
        for j in list(indices):
            iou = compute_iou(boxes[max_idx], boxes[j])
            scores[j] *= np.exp(-iou ** 2 / sigma)

            # 低于阈值的删除
            if scores[j] < score_threshold:
                indices.remove(j)

    return keep, scores


def compute_average_precision(scores: List[float],
                              labels: List[int],
                              num_gt: int,
                              n_points: int = 11) -> float:
    """计算单类AP（11点插值法）

    参数：
        scores: 预测置信度
        labels: 与scores等长，1=TP, 0=FP
        num_gt: 该类真实样本总数
        n_points: 插值点数（VOC=11, COCO=101）
    """
    if num_gt == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    labels = np.array(labels)[order]

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    recall = tp / num_gt
    precision = tp / (tp + fp + 1e-9)

    ap = 0.0
    for t in np.linspace(0, 1, n_points):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / n_points

    return ap


def demo_iou():
    """演示IoU的几种边界情况"""
    print("=== IoU 测试用例 ===")
    test_cases = [
        ([0, 0, 10, 10], [0, 0, 10, 10], "完全重合"),
        ([0, 0, 10, 10], [20, 20, 30, 30], "完全分离"),
        ([0, 0, 10, 10], [5, 5, 15, 15], "对角重叠"),
        ([0, 0, 10, 10], [2, 2, 8, 8], "小框被包含"),
        ([0, 0, 100, 100], [50, 0, 150, 100], "水平错开半幅"),
        ([0, 0, 10, 10], [10, 0, 20, 10], "刚好相切"),
    ]
    for b1, b2, desc in test_cases:
        iou = compute_iou(b1, b2)
        print(f"  {desc:25s}  IoU = {iou:.4f}")


def demo_nms():
    """演示NMS，并可视化前后对比"""
    print("\n=== NMS 测试 ===")
    # 6个框，部分高度重叠
    boxes = np.array([
        [10, 10, 50, 50],     # 框0
        [12, 12, 52, 52],     # 框1 (与0高度重叠)
        [60, 60, 100, 100],   # 框2
        [62, 62, 102, 102],   # 框3 (与2高度重叠)
        [120, 10, 160, 50],   # 框4
        [180, 80, 220, 120],  # 框5
    ])
    scores = np.array([0.95, 0.85, 0.90, 0.80, 0.75, 0.60])

    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"原始6个框，NMS后保留 {len(keep)} 个: {keep}")
    for i in keep:
        print(f"  框{i}: bbox={boxes[i].tolist()}, score={scores[i]:.2f}")

    # 可视化
    canvas_size = (240, 280, 3)
    before = np.full(canvas_size, 255, dtype=np.uint8)
    after = np.full(canvas_size, 255, dtype=np.uint8)
    np.random.seed(0)
    colors = [tuple(int(c) for c in np.random.randint(0, 200, 3))
              for _ in range(len(boxes))]
    for i, (b, s) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = b.tolist()
        cv2.rectangle(before, (x1, y1), (x2, y2), colors[i], 2)
        cv2.putText(before, f"{s:.2f}", (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    for i in keep:
        x1, y1, x2, y2 = boxes[i].tolist()
        cv2.rectangle(after, (x1, y1), (x2, y2), colors[i], 2)
        cv2.putText(after, f"{scores[i]:.2f}", (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    show_images(before, after,
                titles=[f"NMS 前 ({len(boxes)} 个框)",
                        f"NMS 后 ({len(keep)} 个框)"],
                window_size=(10, 5))
    return boxes, scores, keep


def demo_ap():
    """演示AP计算，并绘制PR曲线"""
    print("\n=== AP 计算 ===")
    scores = [0.95, 0.91, 0.88, 0.80, 0.75, 0.60, 0.40]
    labels = [1, 1, 0, 1, 1, 0, 1]    # 5个TP, 2个FP
    num_gt = 5

    ap = compute_average_precision(scores, labels, num_gt, n_points=11)
    print(f"11点插值AP: {ap:.4f}")

    ap_coco = compute_average_precision(scores, labels, num_gt, n_points=101)
    print(f"101点插值AP（COCO标准）: {ap_coco:.4f}")

    # 绘制 PR 曲线
    order = np.argsort(scores)[::-1]
    arr_labels = np.array(labels)[order]
    tp = np.cumsum(arr_labels)
    fp = np.cumsum(1 - arr_labels)
    recall = tp / num_gt
    precision = tp / (tp + fp + 1e-9)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, 'o-', linewidth=2, label='PR 曲线')
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall（召回率）")
    ax.set_ylabel("Precision（准确率）")
    ax.set_title(f"PR 曲线 (AP={ap:.3f}, COCO AP={ap_coco:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/pr_curve.png", dpi=100)
    show_plot()


if __name__ == "__main__":
    demo_iou()
    demo_nms()
    demo_ap()
