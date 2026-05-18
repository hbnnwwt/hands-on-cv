"""第9章：OMR 完整Pipeline

教材对应：第9章 光学标记识别（OMR）

支持命令行：
    python omr_pipeline.py path/to/answer_sheet.png
"""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    load_answer_gt,
    find_gt_for_student_id,
    read_image,
    show_images,
    list_student_ids_from_segmented,
    find_segmented_image_for_id,
)


@dataclass
class BubbleROI:
    """单个气泡的位置和元数据"""
    question_id: int
    option: str           # 'A', 'B', 'C', 'D'
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)


@dataclass
class OMRAnswer:
    """单道选择题的识别结果"""
    question_id: int
    selected: List[str]
    confidence: float
    fill_ratios: dict
    is_abnormal: bool = False
    abnormal_reason: str = ""


def generate_grid(area: dict, num_questions: int = 20,
                  options_per_row: int = 4,
                  questions_per_row: int = 4,
                  bubble_size: int = 40,
                  option_start_ratio: float = 0.345,
                  option_step_ratio: float = 0.182) -> List[BubbleROI]:
    """根据答题卡区域生成所有气泡坐标（选择题）

    option_start_ratio / option_step_ratio 控制每题内 4 个选项气泡的
    水平位置（相对每题宽度）。默认值基于 data/answer_sheet/segmented/
    分割图实测气泡位置校准得出。
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

        option_start_x = q_x + h_per_question * option_start_ratio
        option_spacing = h_per_question * option_step_ratio

        for opt_idx, opt_letter in enumerate(['A', 'B', 'C', 'D'][:options_per_row]):
            bubble_cx = option_start_x + opt_idx * option_spacing
            bubble_cy = q_y + v_per_row / 2

            bubbles.append(BubbleROI(
                question_id=q + 1,
                option=opt_letter,
                bbox=(
                    int(bubble_cx - bubble_size / 2),
                    int(bubble_cy - bubble_size / 2),
                    bubble_size,
                    bubble_size,
                ),
            ))

    return bubbles


def generate_judge_grid(area: dict, num_questions: int = 10,
                        questions_per_row: int = 4,
                        question_start: int = 21,
                        bubble_size: int = 40,
                        option_start_ratio: float = 0.45,
                        option_step_ratio: float = 0.33) -> List[BubbleROI]:
    """根据答题卡区域生成判断题气泡坐标（T/F选项）

    option_start_ratio / option_step_ratio 控制每题内 T/F 气泡的
    水平位置（相对每题宽度）。判断题模板为 "N. □T □F"，
    T 圆约在题宽 0.45 处，F 圆约在 0.78 处。
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

        option_start_x = q_x + h_per_question * option_start_ratio
        option_spacing = h_per_question * option_step_ratio

        for opt_idx, opt_label in enumerate(options):
            bubble_cx = option_start_x + opt_idx * option_spacing
            bubble_cy = q_y + v_per_row / 2

            bubbles.append(BubbleROI(
                question_id=question_start + q,
                option=opt_label,
                bbox=(
                    int(bubble_cx - bubble_size / 2),
                    int(bubble_cy - bubble_size / 2),
                    bubble_size,
                    bubble_size,
                ),
            ))

    return bubbles


def _detect_blobs(roi: np.ndarray) -> List[Tuple[float, float, float]]:
    """在ROI中检测圆形气泡，返回 (cx, cy, area) 列表（ROI局部坐标）

    过滤条件：圆形（宽高比 >= 0.7），尺寸 >= 20px，面积 >= 200。
    同时捕获小圆圈轮廓和填涂后的大连通域。
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    blobs = []
    for i in range(1, num_labels):
        sx, sy, sw, sh, a = stats[i]
        aspect = sw / sh if sh > 0 else 0
        if sw >= 20 and sh >= 20 and 0.7 <= aspect <= 2.0 and a >= 200:
            blobs.append((float(centroids[i][0]), float(centroids[i][1]), float(a)))
    return blobs


def _cluster_x_columns(blobs: List[Tuple[float, float, float]],
                       n_cols: int, threshold: float = 50
                       ) -> List[float]:
    """将所有 blob 的 x 坐标聚类成 n_cols 列，返回排序后的列中心"""
    all_x = sorted(b[0] for b in blobs)
    groups: List[List[float]] = [[all_x[0]]]
    for x in all_x[1:]:
        if x - np.mean(groups[-1]) < threshold:
            groups[-1].append(x)
        else:
            groups.append([x])

    centers = sorted(np.mean(g) for g in groups)
    return centers[:n_cols]


def detect_choice_bubbles(image: np.ndarray, area: dict,
                          num_questions: int = 20,
                          questions_per_row: int = 4,
                          options_per_question: int = 4,
                          bubble_size: int = 70) -> List[BubbleROI]:
    """从图像中自动检测选择题气泡位置（连通域定位 + fill_ratio 判定）

    通过二值化+连通域检测找到气泡位置，聚类 x 坐标建立列结构，
    用每行 blob 的 y 中位数确定行中心。
    最终生成大 ROI 的 BubbleROI 供 fill_ratio 计算用。
    检测失败时自动回退到 generate_grid。
    """
    x0, y0 = area['x'], area['y']
    roi = image[y0:y0 + area['h'], x0:x0 + area['w']]
    blobs = _detect_blobs(roi)

    n_cols = questions_per_row * options_per_question
    if len(blobs) < n_cols:
        return generate_grid(area, num_questions, options_per_question,
                             questions_per_row, bubble_size)

    col_centers = _cluster_x_columns(blobs, n_cols)
    if len(col_centers) < n_cols:
        return generate_grid(area, num_questions, options_per_question,
                             questions_per_row, bubble_size)

    num_rows = (num_questions + questions_per_row - 1) // questions_per_row
    v_per_row = area['h'] / num_rows

    # 按行分组，计算每行的 y 中心
    row_cy = []
    for ri in range(num_rows):
        y_lo, y_hi = ri * v_per_row, (ri + 1) * v_per_row
        row_blobs = [b for b in blobs
                     if y_lo <= b[1] < y_hi
                     and any(abs(b[0] - cc) < 50 for cc in col_centers)]
        row_cy.append(float(np.median([b[1] for b in row_blobs]))
                       if row_blobs else float((y_lo + y_hi) / 2))

    # 为每个 (行, 列) 位置生成 BubbleROI
    bubbles = []
    for ri in range(num_rows):
        for q_col in range(questions_per_row):
            q_id = ri * questions_per_row + q_col + 1
            if q_id > num_questions:
                break
            for oi in range(options_per_question):
                col_idx = q_col * options_per_question + oi
                cx = col_centers[col_idx]
                cy = row_cy[ri]
                opt = chr(ord('A') + oi)
                bubbles.append(BubbleROI(
                    question_id=q_id, option=opt,
                    bbox=(int(cx + x0 - bubble_size / 2),
                          int(cy + y0 - bubble_size / 2),
                          bubble_size, bubble_size),
                ))

    return bubbles


def detect_judge_bubbles(image: np.ndarray, area: dict,
                         num_questions: int = 10,
                         questions_per_row: int = 4,
                         options_per_question: int = 2,
                         question_start: int = 21,
                         bubble_size: int = 70) -> List[BubbleROI]:
    """从图像中自动检测判断题气泡位置（T/F 选项）

    与 detect_choice_bubbles 原理相同：连通域定位列结构，
    用大 ROI 的 BubbleROI 供 fill_ratio 判定。
    """
    x0, y0 = area['x'], area['y']
    roi = image[y0:y0 + area['h'], x0:x0 + area['w']]
    blobs = _detect_blobs(roi)

    n_cols = questions_per_row * options_per_question
    if len(blobs) < n_cols:
        return generate_judge_grid(area, num_questions, questions_per_row,
                                   question_start, bubble_size)

    col_centers = _cluster_x_columns(blobs, n_cols)
    if len(col_centers) < n_cols:
        return generate_judge_grid(area, num_questions, questions_per_row,
                                   question_start, bubble_size)

    num_rows = (num_questions + questions_per_row - 1) // questions_per_row
    v_per_row = area['h'] / num_rows
    option_labels = ['T', 'F']

    # 按行分组，计算每行的 y 中心
    row_cy = []
    for ri in range(num_rows):
        y_lo, y_hi = ri * v_per_row, (ri + 1) * v_per_row
        row_blobs = [b for b in blobs
                     if y_lo <= b[1] < y_hi
                     and any(abs(b[0] - cc) < 50 for cc in col_centers)]
        row_cy.append(float(np.median([b[1] for b in row_blobs]))
                       if row_blobs else float((y_lo + y_hi) / 2))

    bubbles = []
    for ri in range(num_rows):
        for q_col in range(questions_per_row):
            q_id = question_start + ri * questions_per_row + q_col
            if q_id >= question_start + num_questions:
                break
            for oi in range(options_per_question):
                col_idx = q_col * options_per_question + oi
                cx = col_centers[col_idx]
                cy = row_cy[ri]
                bubbles.append(BubbleROI(
                    question_id=q_id, option=option_labels[oi],
                    bbox=(int(cx + x0 - bubble_size / 2),
                          int(cy + y0 - bubble_size / 2),
                          bubble_size, bubble_size),
                ))

    return bubbles


def compute_fill_ratio(roi: np.ndarray) -> float:
    """计算ROI的填涂密度

    返回：0.0~1.0，前景像素占总像素的比例
    """
    if roi.size == 0:
        return 0.0

    # 灰度化（如果是彩色）
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # 二值化（前景为黑色）
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 计算前景像素比例
    return float(np.sum(binary > 0)) / binary.size


class OMRPipeline:
    """OMR 主流程类"""

    def __init__(self,
                 fill_threshold: float = 0.40,
                 multi_select_threshold: float = 0.40,
                 confidence_min: float = 0.20):
        self.fill_threshold = fill_threshold
        self.multi_select_threshold = multi_select_threshold
        self.confidence_min = confidence_min

    def process(self, image: np.ndarray,
                bubbles: List[BubbleROI]) -> List[OMRAnswer]:
        """主处理流程：对每道题判断填涂

        参数：
            image: 答题卡图像（已矫正）
            bubbles: 所有气泡的坐标

        返回：
            每题的OMRAnswer
        """
        # 按题号分组
        question_groups = {}
        for bubble in bubbles:
            q_id = bubble.question_id
            question_groups.setdefault(q_id, []).append(bubble)

        answers = []
        for q_id, q_bubbles in sorted(question_groups.items()):
            answer = self._process_question(image, q_bubbles)
            answers.append(answer)

        return answers

    def _process_question(self, image: np.ndarray,
                          bubbles: List[BubbleROI]) -> OMRAnswer:
        """处理单道选择题"""
        fill_ratios = {}
        h, w = image.shape[:2]

        for bubble in bubbles:
            x, y, bw, bh = bubble.bbox
            # 边界检查
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            if x2 <= x or y2 <= y:
                fill_ratios[bubble.option] = 0.0
                continue

            roi = image[y:y2, x:x2]
            fill_ratios[bubble.option] = compute_fill_ratio(roi)

        # 找出超过阈值的选项
        selected = [opt for opt, ratio in fill_ratios.items()
                    if ratio >= self.fill_threshold]

        # 计算置信度（最高填涂比例与第二高的差距）
        sorted_ratios = sorted(fill_ratios.values(), reverse=True)
        if len(sorted_ratios) >= 2:
            confidence = float(sorted_ratios[0] - sorted_ratios[1])
        else:
            confidence = float(sorted_ratios[0]) if sorted_ratios else 0.0

        # 异常检测
        is_abnormal = False
        abnormal_reason = ""

        if not selected:
            is_abnormal = True
            abnormal_reason = "未检测到填涂"
        elif len(selected) > 1:
            is_abnormal = True
            abnormal_reason = f"多选: {selected}"
        elif confidence < self.confidence_min:
            is_abnormal = True
            abnormal_reason = f"置信度过低: {confidence:.2f}"

        return OMRAnswer(
            question_id=bubbles[0].question_id,
            selected=selected,
            confidence=confidence,
            fill_ratios=fill_ratios,
            is_abnormal=is_abnormal,
            abnormal_reason=abnormal_reason,
        )


def visualize_results(image: np.ndarray,
                      bubbles: List[BubbleROI],
                      answers: List[OMRAnswer]) -> np.ndarray:
    """可视化OMR结果"""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 建立 answer 查询表
    answer_map = {a.question_id: a for a in answers}

    for bubble in bubbles:
        x, y, w, h = bubble.bbox
        answer = answer_map.get(bubble.question_id)

        if answer is None:
            color = (128, 128, 128)
        elif answer.is_abnormal:
            color = (0, 165, 255)  # 橙色：异常
        elif bubble.option in answer.selected:
            color = (0, 255, 0)    # 绿色：已选
        else:
            color = (200, 200, 200)  # 灰色：未选

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, bubble.option, (x + 5, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis


def _find_page2(page1_path: Path) -> Optional[Path]:
    """根据第一页路径查找配对的第二页（test_set_01 ↔ test_set_02）"""
    stem = page1_path.stem
    for i in range(len(stem) - 1, -1, -1):
        if stem[i].isdigit():
            digit = int(stem[i])
            if digit % 2 == 1:  # 奇数 → 偶数
                paired = stem[:i] + str(digit + 1) + stem[i + 1:]
                page2 = page1_path.parent / (paired + page1_path.suffix)
                return page2 if page2.exists() else None
            break
    return None


def main():
    # ---- 确定学号：从参数或 segmented 文件夹自动发现 ----
    available_ids = list_student_ids_from_segmented()
    if not available_ids:
        print("错误：segmented 目录中没有找到分割图")
        print("请先运行版面分割，生成 {学号}_choice.png / {学号}_judge.png")
        sys.exit(1)

    if len(sys.argv) >= 2:
        student_id = sys.argv[1]
    else:
        student_id = available_ids[0]
        print(f"未指定学号，使用第一个: {student_id}")

    if student_id not in available_ids:
        print(f"错误：学号 {student_id} 在 segmented 目录中不存在")
        print(f"可用学号: {available_ids}")
        sys.exit(1)

    print(f"学号: {student_id}")
    Path("outputs").mkdir(exist_ok=True)

    # ---- GT：用学号反查 ----
    all_gt = {}
    gt_files = find_gt_for_student_id(student_id)
    for gf in gt_files:
        all_gt.update(load_answer_gt(gf))
        print(f"已加载GT: {gf.name} ({len(all_gt)}题)")

    pipeline = OMRPipeline(fill_threshold=0.40)
    all_answers: List[OMRAnswer] = []

    # ---- 选择题：直接读取分割图 ----
    choice_seg = find_segmented_image_for_id(student_id, 'choice')
    if choice_seg is None:
        print(f"错误：未找到学号 {student_id} 的选择题分割图")
        sys.exit(1)

    choice_image = read_image(choice_seg)
    ch, cw = choice_image.shape[:2]
    print(f"选择题分割图: {choice_seg.name}, 尺寸={choice_image.shape}")
    choice_area = {
        "x": 0,
        "y": int(ch * 0.328),
        "w": cw,
        "h": int(ch * 0.627),
    }

    choice_bubbles = detect_choice_bubbles(choice_image, choice_area,
                                            num_questions=20)
    print(f"选择题：检测到 {len(choice_bubbles)} 个气泡坐标")
    choice_answers = pipeline.process(choice_image, choice_bubbles)
    all_answers.extend(choice_answers)
    vis_choice = visualize_results(choice_image, choice_bubbles, choice_answers)

    # ---- 判断题：直接读取分割图 ----
    judge_seg = find_segmented_image_for_id(student_id, 'judge')
    vis_judge = None
    if judge_seg is not None:
        judge_image = read_image(judge_seg)
        jh, jw = judge_image.shape[:2]
        print(f"判断题分割图: {judge_seg.name}, 尺寸={judge_image.shape}")
        judge_area = {
            "x": 0,
            "y": int(jh * 0.437),
            "w": jw,
            "h": int(jh * 0.495),
        }
        judge_bubbles = detect_judge_bubbles(judge_image, judge_area,
                                              num_questions=10,
                                              question_start=21)
        print(f"判断题：检测到 {len(judge_bubbles)} 个气泡坐标")
        judge_answers = pipeline.process(judge_image, judge_bubbles)
        all_answers.extend(judge_answers)
        vis_judge = visualize_results(judge_image, judge_bubbles, judge_answers)
        cv2.imwrite("outputs/omr_result_page2.png", vis_judge)
        print("已保存第二页可视化: outputs/omr_result_page2.png")

    # ---- 打印结果 ----
    print(f"\n=== OMR 识别结果（共{len(all_answers)}题）===")
    for ans in all_answers:
        status = "[!]" if ans.is_abnormal else "[OK]"
        print(f"{status} 第{ans.question_id}题: {ans.selected} "
              f"(置信度={ans.confidence:.2f})", end="")
        if ans.is_abnormal:
            print(f"  [{ans.abnormal_reason}]")
        else:
            print()

    cv2.imwrite("outputs/omr_result.png", vis_choice)
    print("\n已保存可视化结果: outputs/omr_result.png")

    with open("outputs/omr_result.json", 'w', encoding='utf-8') as f:
        json.dump([asdict(a) for a in all_answers], f,
                  ensure_ascii=False, indent=2)
    print("已保存JSON结果: outputs/omr_result.json")

    # ---- GT 对比 ----
    if all_gt:
        correct = 0
        total = 0
        for ans in all_answers:
            q_key = str(ans.question_id)
            if q_key not in all_gt:
                continue
            total += 1
            expected = all_gt[q_key]
            detected = ''.join(sorted(ans.selected)) if ans.selected else '-'
            is_correct = detected == expected
            if is_correct:
                correct += 1
            else:
                print(f"  [X] 第{q_key}题: 检测={detected}, GT={expected}")
        if total > 0:
            print(f"\n=== GT 对比 ===")
            print(f"  正确: {correct}/{total} ({correct/total*100:.1f}%)")
    else:
        print(f"\n未找到GT文件，跳过对比")

    # ---- 可视化弹窗 ----
    if vis_judge is not None:
        n_choice = sum(1 for a in all_answers if a.question_id <= 20)
        n_judge = sum(1 for a in all_answers if a.question_id > 20)
        show_images(vis_choice, vis_judge,
                    titles=[f"选择题OMR（{n_choice}题）",
                            f"判断题OMR（{n_judge}题）"],
                    window_size=(14, 7))
    else:
        show_images(vis_choice,
                    titles=[f"选择题OMR（{len(all_answers)}题）"],
                    window_size=(12, 6))


if __name__ == "__main__":
    main()
