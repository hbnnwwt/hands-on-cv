"""第14章：手写文字推理、解码与阅卷后处理

教材对应：
  §14.3 TrOCR 推理
  §14.6.2 评价指标（CER / WER）
  §14.6.6 CTC 贪心解码 + 缩放因子对比
  §14.7 阅卷场景（整题识别、关键词评分、人工复审）
  §14.8 调试技巧（OOM、乱码、中文识别）

运行依赖：
  pip install torch transformers pillow opencv-python numpy
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ═══════════════════════════════════════════
# §14.6.3 TrOCR 单图推理
# ═══════════════════════════════════════════


class TrOCRRecognizer:
    """TrOCR 推理封装：加载模型 → 预处理 → generate → 解码"""

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def recognize(self, image_path: str, max_length: int = 100) -> str:
        """识别一张手写图片"""
        image = Image.open(image_path).convert("RGB")
        return self.recognize_pil(image, max_length)

    def recognize_pil(self, image: Image.Image, max_length: int = 100) -> str:
        """识别 PIL Image 对象"""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=max_length)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def recognize_batch(self, images: List[Image.Image], max_length: int = 100,
                        batch_size: int = 4) -> List[str]:
        """批量推理（控制 batch_size 避免 OOM）"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            pixel_values = self.processor(batch, return_tensors="pt", padding=True).pixel_values
            pixel_values = pixel_values.to(self.device)
            with torch.no_grad():
                gen_ids = self.model.generate(pixel_values, max_length=max_length)
            texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
            results.extend(texts)
        return results


# ═══════════════════════════════════════════
# §14.6.6 CTC 贪心解码
# ═══════════════════════════════════════════


def ctc_greedy_decode(probs: np.ndarray, blank_id: int = 0,
                       idx2char: dict | None = None) -> str:
    """CTC 贪心解码：argmax → 合并相邻重复 → 去除 blank

    参数:
        probs: (T, C+1) 各时间步字符概率
        blank_id: blank token 的 id
        idx2char: id → 字符映射
    """
    preds = np.argmax(probs, axis=-1)
    result = []
    prev = -1
    for p in preds:
        if p != prev and p != blank_id:
            result.append(idx2char[p] if idx2char else str(p))
        prev = p
    return "".join(result)


# ═══════════════════════════════════════════
# §14.7.1 简答题整题识别
# ═══════════════════════════════════════════


def recognize_essay_answer(image: np.ndarray, line_regions: list,
                           recognizer: TrOCRRecognizer) -> str:
    """识别简答题答案：逐行裁剪 → 预处理 → TrOCR 识别 → 拼接

    参数:
        image: 答题卡图像（已矫正）
        line_regions: 文本行区域列表，每个元素包含 bbox (x, y, w, h)
        recognizer: TrOCRRecognizer 实例
    返回:
        完整答案文本
    """
    # 预处理函数（从 trocr_finetune 导入也可）
    def _preprocess(img):
        from trocr_finetune import preprocess_handwriting, resize_keep_aspect
        binary = preprocess_handwriting(img)
        # TrOCR 需要 RGB 三通道输入
        rgb = cv2.cvtColor(resize_keep_aspect(binary), cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    lines_text = []
    for region in line_regions:
        x, y, w, h = region["bbox"] if isinstance(region, dict) else region.bbox
        line_image = image[y: y + h, x: x + w]
        processed = _preprocess(line_image)
        text = recognizer.recognize_pil(processed)
        lines_text.append(text.strip())
    return "\n".join(lines_text)


# ═══════════════════════════════════════════
# §14.7.2 关键词匹配评分
# ═══════════════════════════════════════════


def grade_essay(predicted_text: str, keywords: List[str],
                weights: List[float], max_score: float) -> float:
    """关键词匹配评分，支持模糊匹配（容错 OCR 小错误）

    参数:
        predicted_text: OCR 识别出的文本
        keywords: 标准答案关键词列表
        weights: 每个关键词的权重
        max_score: 满分
    返回:
        得分
    """
    text_lower = predicted_text.lower()
    score = 0.0
    for kw, weight in zip(keywords, weights):
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            score += weight
        else:
            # 模糊匹配：滑窗比对
            best_ratio = 0.0
            for i in range(len(text_lower) - len(kw_lower) + 1):
                ratio = SequenceMatcher(
                    None, kw_lower, text_lower[i: i + len(kw_lower)]
                ).ratio()
                best_ratio = max(best_ratio, ratio)
            if best_ratio >= 0.8:
                score += weight * 0.5  # 部分给分
    return min(score, max_score)


# ═══════════════════════════════════════════
# §14.7.5 人工复审判断
# ═══════════════════════════════════════════


class EssayResult:
    """识别结果容器"""
    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence


def needs_review(essay_result: EssayResult, threshold: float = 0.85) -> bool:
    """判断是否需要人工复审"""
    # 1. 置信度过低
    if essay_result.confidence < threshold:
        return True
    # 2. 文本过短或过长
    if len(essay_result.text) < 5 or len(essay_result.text) > 200:
        return True
    # 3. 含特殊字符（OCR 乱码特征）
    if re.search(r"[^一-鿿a-zA-Z0-9，。,.\s]+", essay_result.text):
        return True
    return False


# ═══════════════════════════════════════════
# §14.8 调试工具函数
# ═══════════════════════════════════════════


def safe_generate(model, processor, pixel_values, max_length=64,
                  num_beams=4, half_precision=False):
    """安全推理：防止 OOM + 防止重复/乱码

    整合 §14.8.1 ~ §14.8.2 的调试建议：
    - 半精度推理省显存
    - 正确设置生成参数防止重复
    """
    if half_precision:
        model.half()
        pixel_values = pixel_values.half()

    generated_ids = model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        early_stopping=True,
        length_penalty=1.0,
        repetition_penalty=1.2,
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def debug_tokens(model, processor, pixel_values, max_length=20):
    """调试工具：打印 token id / token / text，排查乱码"""
    gen_ids = model.generate(pixel_values, max_length=max_length, num_beams=1)
    for i, ids in enumerate(gen_ids):
        print(f"[样本 {i}]")
        print(f"  Token IDs : {ids.tolist()}")
        print(f"  Tokens    : {processor.tokenizer.convert_ids_to_tokens(ids)}")
        print(f"  Text      : {processor.batch_decode([ids], skip_special_tokens=True)[0]}")


# ═══════════════════════════════════════════
# §14.6.6 缩放因子对比演示（NumPy）
# ═══════════════════════════════════════════


def demo_ctc_decode():
    """CTC 贪心解码示例"""
    def softmax_np(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    np.random.seed(7)
    T, C = 8, 4  # 4 字符 + 1 blank (id=0)
    probs = softmax_np(np.random.randn(T, C + 1), axis=-1)
    idx2char = {1: "a", 2: "b", 3: "c", 4: "d"}
    result = ctc_greedy_decode(probs, blank_id=0, idx2char=idx2char)
    print(f"CTC 贪心解码结果: {result}")
    print(f"  probs shape: {probs.shape}, argmax 路径: {np.argmax(probs, axis=-1).tolist()}")


def demo_scale_factor():
    """对比有无 sqrt(d_k) 缩放时的 softmax 分布"""
    def softmax_np(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    np.random.seed(0)
    print(f"{'d_k':>6}  {'no_scale_max':>14}  {'scaled_max':>12}")
    for d_k in [16, 64, 256, 1024]:
        q = np.random.randn(d_k)
        K = np.random.randn(10, d_k)
        w_no = softmax_np(K @ q)
        w_yes = softmax_np(K @ q / np.sqrt(d_k))
        print(f"{d_k:6d}  {w_no.max():14.3f}  {w_yes.max():12.3f}")


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("§14.6.6 CTC 贪心解码演示")
    print("=" * 50)
    demo_ctc_decode()

    print("\n" + "=" * 50)
    print("§14.6.6 缩放因子对比")
    print("=" * 50)
    demo_scale_factor()

    print("\n" + "=" * 50)
    print("§14.6.3 TrOCR 推理示例")
    print("=" * 50)
    print("用法:")
    print("  recognizer = TrOCRRecognizer()")
    print('  text = recognizer.recognize("handwriting.jpg")')
    print("  print(text)")
    print()
    print("§14.7.2 关键词评分示例:")
    print('  grade_essay("光合作用产生氧气", ["光合作用","氧气"], [5, 5], 10)')

    # 实际运行评分演示
    score = grade_essay(
        "光合作用产生氧气",
        keywords=["光合作用", "氧气"],
        weights=[5.0, 5.0],
        max_score=10.0,
    )
    print(f"  -> 得分: {score}")

    # 复审判断演示
    result = EssayResult(text="光合作用产生氧气", confidence=0.9)
    print(f"\n§14.7.5 人工复审判断: needs_review = {needs_review(result)}")
    result_bad = EssayResult(text="???@#$", confidence=0.6)
    print(f"  (乱码样本) needs_review = {needs_review(result_bad)}")
