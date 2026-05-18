"""第18章：单元测试示例

教材对应：第18章 18.2 单元测试

运行：
    pytest chapter18/test_omr.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import cv2

# 添加父目录到path以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from chapter09.omr_pipeline import (
    OMRPipeline, BubbleROI, OMRAnswer, compute_fill_ratio
)


@pytest.fixture
def omr():
    """创建OMR实例的fixture"""
    return OMRPipeline(fill_threshold=0.30, confidence_min=0.5)


@pytest.fixture
def filled_bubble_image():
    """生成一张包含填涂气泡的测试图"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.circle(img, (50, 50), 18, (0, 0, 0), -1)  # 实心黑色圆
    return img


@pytest.fixture
def empty_bubble_image():
    """生成一张空气泡的测试图"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.circle(img, (50, 50), 18, (0, 0, 0), 2)  # 空心黑色圆
    return img


def test_compute_fill_ratio_filled(filled_bubble_image):
    """完全填涂的气泡应该有高密度"""
    roi = filled_bubble_image[30:70, 30:70]
    ratio = compute_fill_ratio(roi)
    assert ratio > 0.5, f"填涂气泡的密度应该高，实际: {ratio}"


def test_compute_fill_ratio_empty(empty_bubble_image):
    """空气泡（只有边框）应该有低密度"""
    roi = empty_bubble_image[30:70, 30:70]
    ratio = compute_fill_ratio(roi)
    assert ratio < 0.4, f"空气泡的密度应该低，实际: {ratio}"


def test_compute_fill_ratio_blank():
    """全白图像密度应为0"""
    blank = np.ones((40, 40, 3), dtype=np.uint8) * 255
    ratio = compute_fill_ratio(blank)
    assert ratio < 0.05


def test_compute_fill_ratio_black():
    """全黑图像密度应为1"""
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    ratio = compute_fill_ratio(black)
    assert ratio > 0.95


def test_omr_single_choice(omr, filled_bubble_image):
    """测试单选题正确识别"""
    bubbles = [
        BubbleROI(question_id=1, option='A', bbox=(30, 30, 40, 40)),
    ]
    answers = omr.process(filled_bubble_image, bubbles)
    assert len(answers) == 1
    assert 'A' in answers[0].selected


def test_omr_no_selection(omr, empty_bubble_image):
    """测试漏选检测"""
    bubbles = [
        BubbleROI(question_id=1, option='A', bbox=(30, 30, 40, 40)),
        BubbleROI(question_id=1, option='B', bbox=(70, 30, 40, 40)),
    ]
    # 把右侧也设为空白
    larger = np.ones((100, 120, 3), dtype=np.uint8) * 255
    cv2.circle(larger, (50, 50), 18, (0, 0, 0), 2)  # 左侧空心圆
    cv2.circle(larger, (90, 50), 18, (0, 0, 0), 2)  # 右侧空心圆

    answers = omr.process(larger, bubbles)
    assert answers[0].is_abnormal
    assert "未检测到填涂" in answers[0].abnormal_reason


def test_omr_bbox_out_of_bounds(omr):
    """超出图像边界的bbox应该被安全处理"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bubbles = [
        BubbleROI(question_id=1, option='A', bbox=(90, 90, 50, 50)),
    ]
    # 不应该崩溃
    answers = omr.process(img, bubbles)
    assert len(answers) == 1


@pytest.mark.parametrize("fill_threshold,is_filled,expected_selected", [
    (0.2, True, True),    # 低阈值，应识别为填涂
    (0.4, True, True),
    (0.6, False, False),  # 高阈值，半涂被识别为未填涂
])
def test_omr_threshold_effect(fill_threshold, is_filled, expected_selected):
    """测试不同阈值对识别结果的影响"""
    omr = OMRPipeline(fill_threshold=fill_threshold)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255

    if is_filled:
        # 半涂（约50%密度）
        cv2.circle(img, (50, 50), 18, (0, 0, 0), -1)
        # 用白色覆盖一半
        cv2.rectangle(img, (50, 30), (70, 70), (255, 255, 255), -1)

    bubbles = [BubbleROI(1, 'A', (30, 30, 40, 40))]
    answers = omr.process(img, bubbles)
    selected = bool(answers[0].selected)
    assert selected == expected_selected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
