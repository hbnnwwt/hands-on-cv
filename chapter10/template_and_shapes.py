"""第10章：模板匹配与形状识别

教材对应：第10章 10.2 模板匹配实战、10.4 轮廓特征匹配
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_images


def template_matching_demo(image_path: str, template_path: str = None):
    """模板匹配演示，返回 (image, template, vis) 用于可视化"""
    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"无法读取 {image_path}: {e}")
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 如果没提供模板，从图像中切一块作为模板
    if template_path is None or not Path(template_path).exists():
        h, w = gray.shape
        template = gray[h // 3:h // 3 + 50, w // 3:w // 3 + 50]
        print("使用图像左上区域作为模板")
    else:
        try:
            template = read_image(template_path, grayscale=True)
        except (FileNotFoundError, ValueError) as e:
            print(f"无法读取模板 {template_path}: {e}")
            return image, None, None

    th, tw = template.shape

    # 6种匹配方法
    methods = [
        ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
    ]

    vis = image.copy()
    colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]

    for (name, method), color in zip(methods, colors):
        result = cv2.matchTemplate(gray, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # SQDIFF是最小值最佳，其他是最大值最佳
        if method == cv2.TM_SQDIFF_NORMED:
            top_left = min_loc
            score = min_val
        else:
            top_left = max_loc
            score = max_val

        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(vis, top_left, bottom_right, color, 2)
        cv2.putText(vis, f"{name.replace('TM_', '')}:{score:.2f}",
                    (top_left[0], max(top_left[1] - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        print(f"{name}: 位置={top_left}, 分数={score:.4f}")

    return image, template, vis


def compute_hu_moments(image: np.ndarray) -> np.ndarray:
    """计算图像中所有轮廓的Hu矩"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 100:
            continue

        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()

        # 对数变换增强区分度（处理符号问题）
        log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        results.append({
            'index': i,
            'area': cv2.contourArea(cnt),
            'hu_moments': hu.tolist(),
            'log_hu': log_hu.tolist(),
        })

    return results


def shape_matching_demo(image_path: str):
    """演示Hu矩的形状匹配"""
    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError):
        return

    print("\n=== Hu矩形状描述 ===")
    results = compute_hu_moments(image)
    print(f"检测到 {len(results)} 个有效轮廓")

    for r in results[:5]:
        print(f"  轮廓{r['index']}: 面积={r['area']:.0f}")
        print(f"    Hu矩 (log): {[f'{v:.3f}' for v in r['log_hu']]}")


def check_cross_recognition(image_path: str):
    """演示判断题√/×识别，返回 vis"""
    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError):
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    vis = image.copy()
    print("\n=== 形状识别（√/×） ===")
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        # 计算特征
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)

        # 凸性缺陷数（√和×的笔画数不同）
        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(cnt, hull)
                n_defects = len(defects) if defects is not None else 0
            except cv2.error:
                n_defects = 0
        else:
            n_defects = 0

        # 简单判别（实际中需要更复杂的特征）
        # √ 通常有1个明显的转折，× 通常有2个交叉
        aspect = w / h if h > 0 else 0
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

        # 可视化：画轮廓 + 圆度标签
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(vis, f"#{i} C={circularity:.2f}",
                    (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        print(f"轮廓{i}: 位置=({x},{y}), 宽高={w}x{h}, "
              f"圆度={circularity:.3f}, 缺陷数={n_defects}")

    return vis


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "shapes.png")
        if not Path(image_path).exists():
            print("提示：先运行 python common/test_images.py")
            return

    Path("outputs").mkdir(exist_ok=True)

    print("=== 模板匹配 ===")
    image, template, match_vis = template_matching_demo(image_path)
    shape_matching_demo(image_path)
    shape_vis = check_cross_recognition(image_path)

    # 弹窗可视化
    if image is not None and match_vis is not None and shape_vis is not None:
        show_images(image, match_vis, shape_vis,
                    titles=["原图", "模板匹配结果", "形状识别结果"],
                    window_size=(15, 5))


if __name__ == "__main__":
    main()
