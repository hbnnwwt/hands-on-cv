"""第17章：完整智能阅卷系统的简化Pipeline

教材对应：第17章 综合项目：智能阅卷系统

这是项目代码的入口骨架。完整实现请参考 auto_grading 包。
"""

from __future__ import annotations
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import json

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

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """评分结果"""
    student_id: str = ""
    total_score: float = 0.0
    max_score: float = 100.0
    choice_results: List[dict] = field(default_factory=list)
    need_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class AutoGradingPipeline:
    """简化版的智能阅卷Pipeline

    完整版包括：
    - 预处理（chapter05）
    - 透视矫正（chapter06）
    - 版面分析（chapter08）
    - OMR识别（chapter09）
    - 模板匹配（chapter10）
    - OCR（chapter13/14）
    - 评分
    """

    def __init__(self, config: dict):
        self.config = config
        self.answer_key = config.get('answer_key', {})

    def process(self, student_id: str) -> Optional[GradingResult]:
        """处理一名学生的答题卡（从分割图读取）"""
        import time
        start = time.time()

        try:
            logger.info(f"处理学号: {student_id}")

            from chapter09.omr_pipeline import detect_choice_bubbles, detect_judge_bubbles, OMRPipeline

            # 1. 选择题：直接读取分割图
            choice_seg = find_segmented_image_for_id(student_id, 'choice')
            if choice_seg is None:
                logger.error(f"未找到学号 {student_id} 的选择题分割图")
                return None

            logger.info(f"选择题分割图: {choice_seg.name}")
            choice_image = read_image(choice_seg)
            ch, cw = choice_image.shape[:2]
            grid_config = {
                'x': 0,
                'y': int(ch * 0.328),
                'w': cw,
                'h': int(ch * 0.627),
            }
            bubbles = detect_choice_bubbles(choice_image, grid_config)

            # 2. OMR（选择题）
            omr = OMRPipeline(
                fill_threshold=self.config.get('fill_threshold', 0.50)
            )
            choices = omr.process(choice_image, bubbles)

            # 3. 判断题：直接读取分割图
            judge_seg = find_segmented_image_for_id(student_id, 'judge')
            if judge_seg is not None:
                logger.info(f"判断题分割图: {judge_seg.name}")
                judge_image = read_image(judge_seg)
                jh, jw = judge_image.shape[:2]
                judge_config = {
                    'x': 0,
                    'y': int(jh * 0.437),
                    'w': jw,
                    'h': int(jh * 0.495),
                }
                judge_bubbles = detect_judge_bubbles(
                    judge_image, judge_config, num_questions=10,
                    question_start=21)
                judge_answers = omr.process(judge_image, judge_bubbles)
                choices.extend(judge_answers)

            # 4. 评分
            result = self._grade(choices, student_id)
            result.processing_time_ms = (time.time() - start) * 1000

            logger.info(f"完成: 得分={result.total_score}, "
                       f"耗时={result.processing_time_ms:.0f}ms")
            return result

        except Exception as e:
            logger.exception(f"处理失败: {student_id}")
            return None

    def _grade(self, choices, student_id: str) -> GradingResult:
        """根据识别结果和标准答案评分"""
        result = GradingResult(student_id=student_id)

        score_per_question = self.answer_key.get('score_per_question', 5)
        judge_score = self.answer_key.get('judge_score_per_question', 5)
        correct_answers = self.answer_key.get('answers', {})

        for choice in choices:
            q_id = choice.question_id
            correct = correct_answers.get(str(q_id))
            selected = choice.selected

            if correct is None:
                continue

            # 判断题（题号>=21）或选择题：单选完全匹配
            is_correct = (len(selected) == 1 and selected[0] == correct)
            score = (judge_score if q_id >= 21 else score_per_question) if is_correct else 0

            result.total_score += score
            result.choice_results.append({
                'q_id': q_id,
                'correct': correct,
                'selected': selected,
                'score': score,
                'is_correct': is_correct,
                'is_abnormal': choice.is_abnormal,
            })

            if choice.is_abnormal:
                result.need_review = True
                result.review_reasons.append(
                    f"题{q_id}: {choice.abnormal_reason}"
                )

        return result


def render_score_card(result: GradingResult,
                      canvas_size: Tuple[int, int] = (600, 800)) -> np.ndarray:
    """把评分结果渲染为一张文字卡片，用于可视化弹窗"""
    h, w = canvas_size
    card = np.full((h, w, 3), 250, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(card, f"Student: {result.student_id}", (30, 60),
                font, 1.0, (40, 40, 40), 2)
    cv2.putText(card, f"Score: {result.total_score:.1f} / {result.max_score:.0f}",
                (30, 110), font, 1.2, (0, 120, 0), 3)
    cv2.putText(card, f"Time: {result.processing_time_ms:.0f} ms",
                (30, 150), font, 0.6, (80, 80, 80), 1)

    # 题目列表（每行画对错）
    y = 200
    cv2.putText(card, "Q  Selected   Correct  Score", (30, y),
                font, 0.55, (0, 0, 0), 1)
    y += 25
    cv2.line(card, (30, y), (w - 30, y), (200, 200, 200), 1)
    y += 20

    for r in result.choice_results[:25]:  # 最多显示前25题
        color = (0, 150, 0) if r['is_correct'] else (0, 0, 200)
        if r.get('is_abnormal'):
            color = (0, 120, 200)
        sel = ''.join(r['selected']) if r['selected'] else '-'
        line = f"{r['q_id']:>2}  {sel:<8}  {r['correct']:<6}  {r['score']}"
        cv2.putText(card, line, (30, y), font, 0.55, color, 1)
        y += 24
        if y > h - 30:
            break

    # 复核提示
    if result.need_review:
        cv2.putText(card, f"NEED REVIEW ({len(result.review_reasons)})",
                    (30, h - 20), font, 0.6, (0, 0, 200), 2)

    return card


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # 确定学号：从参数或 segmented 文件夹自动发现
    available_ids = list_student_ids_from_segmented()
    if not available_ids:
        print("错误：segmented 目录中没有找到分割图")
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

    # 从 GT 文件加载标准答案（用学号反查）
    all_gt = {}
    gt_files = find_gt_for_student_id(student_id)
    for gf in gt_files:
        all_gt.update(load_answer_gt(gf))
        print(f"已加载GT: {gf.name} ({len(all_gt)}题)")

    if not all_gt:
        print("未找到GT文件，使用空答案")

    config = {
        'fill_threshold': 0.50,
        'answer_key': {
            'score_per_question': 5,
            'judge_score_per_question': 5,
            'answers': all_gt,
        }
    }

    pipeline = AutoGradingPipeline(config)
    result = pipeline.process(student_id)

    if result is None:
        print("处理失败")
        sys.exit(1)

    print("\n=== 评分结果 ===")
    print(f"学生ID: {result.student_id}")
    print(f"总分: {result.total_score}/{result.max_score}")
    print(f"处理耗时: {result.processing_time_ms:.0f}ms")
    print(f"需要复核: {result.need_review}")
    if result.review_reasons:
        print(f"复核原因: {result.review_reasons}")

    # 保存JSON
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/grading_result.json", 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    print("\n结果已保存: outputs/grading_result.json")

    # 可视化评分卡片
    try:
        card = render_score_card(result)
        cv2.imwrite("outputs/grading_card.png", card)
        show_images(card,
                    titles=[f"评分结果（{result.total_score:.0f}分）"],
                    window_size=(10, 8))
    except (FileNotFoundError, ValueError) as e:
        print(f"可视化跳过：{e}")


if __name__ == "__main__":
    main()
