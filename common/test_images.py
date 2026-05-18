"""生成测试图像：当用户没有真实图像时，用代码生成示例数据

这避免了仓库带大量二进制文件。运行此脚本会在 data/ 目录生成多种测试图像。
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import save_image


def generate_blank_paper(width: int = 800, height: int = 1100) -> np.ndarray:
    """生成空白答题卡（白底）"""
    return np.ones((height, width, 3), dtype=np.uint8) * 250


def generate_answer_sheet() -> np.ndarray:
    """生成模拟答题卡（含定位点、信息区、选择题、判断题、简答题）"""
    img = generate_blank_paper(800, 1100)

    # 四角定位点（黑色方块）
    for cy, cx in [(40, 40), (40, 760), (1060, 40), (1060, 760)]:
        cv2.rectangle(img, (cx - 15, cy - 15), (cx + 15, cy + 15), 0, -1)

    # 标题区
    cv2.putText(img, "Auto Grading Sample", (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # 信息区横线
    cv2.line(img, (50, 150), (750, 150), (0, 0, 0), 2)

    # 选择题区域（20题，5行×4题）
    for row in range(5):
        for q_idx in range(4):
            q_num = row * 4 + q_idx + 1
            base_x = 80 + q_idx * 175
            base_y = 220 + row * 80

            # 题号
            cv2.putText(img, f"{q_num}.", (base_x - 30, base_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 4个气泡（A B C D）
            for opt_idx, letter in enumerate(['A', 'B', 'C', 'D']):
                cx = base_x + opt_idx * 35
                cv2.circle(img, (cx, base_y), 10, (0, 0, 0), 2)
                cv2.putText(img, letter, (cx - 5, base_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # 判断题区域
    cv2.line(img, (50, 700), (750, 700), (0, 0, 0), 2)
    for row in range(2):
        for q_idx in range(5):
            q_num = row * 5 + q_idx + 1
            cx = 80 + q_idx * 130
            cy = 740 + row * 60
            cv2.putText(img, f"{q_num}.", (cx - 30, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(img, (cx, cy - 15), (cx + 30, cy + 15), (0, 0, 0), 2)

    # 简答题区域
    cv2.line(img, (50, 900), (750, 900), (0, 0, 0), 2)
    cv2.putText(img, "Essay (write below):", (60, 930),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for y in range(960, 1050, 30):
        cv2.line(img, (80, y), (720, y), (200, 200, 200), 1)

    return img


def generate_filled_answer_sheet() -> np.ndarray:
    """生成已填涂的答题卡（部分气泡填涂）"""
    img = generate_answer_sheet()

    # 填涂部分气泡（示例：第1题选A，第2题选B，等）
    fills = [
        (0, 0), (1, 1), (2, 2), (3, 3),   # 1A, 2B, 3C, 4D
        (4, 0), (5, 1), (6, 2), (7, 3),
    ]
    for q_idx, opt_idx in fills:
        row = q_idx // 4
        col = q_idx % 4
        base_x = 80 + col * 175
        base_y = 220 + row * 80
        cx = base_x + opt_idx * 35
        cv2.circle(img, (cx, base_y), 10, (0, 0, 0), -1)

    # 判断题：第1题√，第2题×
    cv2.line(img, (85, 745), (95, 755), (0, 0, 0), 2)
    cv2.line(img, (95, 755), (108, 720), (0, 0, 0), 2)

    cv2.line(img, (215, 730), (240, 755), (0, 0, 0), 2)
    cv2.line(img, (240, 730), (215, 755), (0, 0, 0), 2)

    return img


def generate_noisy_image(base: np.ndarray | None = None,
                         noise_level: float = 30) -> np.ndarray:
    """在图像上添加高斯噪声"""
    if base is None:
        base = generate_answer_sheet()
    noise = np.random.randn(*base.shape) * noise_level
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_tilted_image(base: np.ndarray | None = None,
                          canvas_size: tuple[int, int] = (1280, 1600),
                          bg_color: tuple[int, int, int] = (60, 60, 60)) -> np.ndarray:
    """生成透视畸变的答题卡图像（模拟相机斜拍）

    把答题卡用 4 点透视变换贴到深灰色背景上，让 Canny 能检测出纸边。
    """
    if base is None:
        base = generate_answer_sheet()
    h, w = base.shape[:2]

    cw, ch = canvas_size
    canvas = np.full((ch, cw, 3), bg_color, dtype=np.uint8)

    # 源点：纸张四角
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # 目标点：在画布上做透视畸变（上窄下宽，右倾）
    margin_x, margin_y = 180, 120
    dst = np.float32([
        [margin_x + 60,           margin_y + 40],          # 左上
        [cw - margin_x - 20,      margin_y + 90],          # 右上（更靠下）
        [cw - margin_x - 60,      ch - margin_y - 60],     # 右下
        [margin_x + 20,           ch - margin_y - 100],    # 左下（更靠上）
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(base, M, (cw, ch),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_TRANSPARENT)

    # 用 mask 把变形后的纸贴到深色背景上
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    canvas[mask > 0] = warped[mask > 0]

    return canvas


def generate_shapes() -> np.ndarray:
    """生成一张包含多种几何形状的测试图（用于第10章模板匹配）"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # 圆
    cv2.circle(img, (100, 100), 40, (0, 0, 0), 2)
    # 三角形
    pts = np.array([[300, 60], [250, 140], [350, 140]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    # 矩形
    cv2.rectangle(img, (450, 60), (550, 140), (0, 0, 0), 2)
    # 五角星
    cv2.polylines(img, [np.array([
        [100, 220], [120, 270], [170, 280],
        [130, 310], [145, 360], [100, 330],
        [55, 360], [70, 310], [30, 280], [80, 270]
    ], np.int32)], True, (0, 0, 0), 2)
    return img


def main():
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "blank_paper.png": generate_blank_paper(),
        "answer_sheet.png": generate_answer_sheet(),
        "filled_answer_sheet.png": generate_filled_answer_sheet(),
        "noisy_paper.png": generate_noisy_image(),
        "tilted_paper.png": generate_tilted_image(),
        "shapes.png": generate_shapes(),
    }

    for name, img in samples.items():
        path = out_dir / name
        save_image(img, path)
        print(f"[OK] 生成 {path}")

    print(f"\n共生成 {len(samples)} 张测试图像于 {out_dir}")


if __name__ == "__main__":
    main()
