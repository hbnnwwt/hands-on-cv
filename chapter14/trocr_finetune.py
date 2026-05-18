"""第14章：TrOCR 微调训练 —— Transformer 基础 + 手写文字识别

教材对应：
  §14.2 Transformer 基础（Self-Attention、多头注意力、位置编码）
  §14.3 TrOCR 架构（ViT 编码器、因果掩码）
  §14.5 数据增强
  §14.6.7 TrOCR 完整微调代码

运行依赖：
  pip install torch transformers pillow pandas tqdm Levenshtein albumentations opencv-python
"""

from __future__ import annotations

import math
import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)

# ═══════════════════════════════════════════
# §14.2.2 Self-Attention（PyTorch 实现）
# ═══════════════════════════════════════════


class SelfAttention(nn.Module):
    """单头自注意力：Q/K/V 线性投影 → 缩放点积 → softmax → 加权求和"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: (B, L, D)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out, attn_weights


# ═══════════════════════════════════════════
# §14.2.3 多头注意力
# ═══════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """多头自注意力：把 D 维拆成 H 个 d_k 子空间并行计算"""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(context)


# ═══════════════════════════════════════════
# §14.2.4 正弦位置编码
# ═══════════════════════════════════════════


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """经典 Transformer 正弦/余弦位置编码 (seq_len, d_model)"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ═══════════════════════════════════════════
# §14.3.2 ViT PatchEmbedding（TrOCR 编码器核心）
# ═══════════════════════════════════════════


class PatchEmbedding(nn.Module):
    """将图像切割为 patch 并投影到 embed_dim 维向量，加位置编码"""

    def __init__(self, image_size: int = 384, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, num_patches, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x


# ═══════════════════════════════════════════
# §14.3.3 因果掩码（解码器用）
# ═══════════════════════════════════════════


def causal_mask(seq_len: int) -> torch.Tensor:
    """下三角因果掩码，防止解码器看到未来位置"""
    return torch.tril(torch.ones(seq_len, seq_len))


# ═══════════════════════════════════════════
# §14.4 手写图像预处理
# ═══════════════════════════════════════════


def remove_grid_lines(gray: np.ndarray) -> np.ndarray:
    """去除横格线（水平直线）"""
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(cv2.bitwise_not(gray), cv2.MORPH_OPEN, kernel_h)
    result = gray.copy()
    result[horizontal > 100] = 255
    return result


def preprocess_handwriting(image: np.ndarray) -> np.ndarray:
    """手写图像预处理 Pipeline：灰度 → 去横线 → CLAHE → 二值化 → 形态学"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = remove_grid_lines(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def deskew_line(line_image: np.ndarray) -> np.ndarray:
    """矫正单行手写文字的倾斜"""
    coords = np.column_stack(np.where(line_image > 0))
    if len(coords) < 10:
        return line_image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    h, w = line_image.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(line_image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def resize_keep_aspect(image: np.ndarray, target_size=(384, 384), pad_color=255):
    """等比例缩放并居中填充到目标尺寸"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    if image.ndim == 3:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[pad_color] * 3)
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_color)
    return padded


# ═══════════════════════════════════════════
# §14.5 数据增强
# ═══════════════════════════════════════════


def stroke_augment(image: np.ndarray, thickness_change: float = 0.3):
    """随机改变笔画粗细（膨胀或腐蚀）"""
    if np.random.random() < 0.5:
        kernel_size = np.random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
    else:
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(image, kernel, iterations=1)


# ═══════════════════════════════════════════
# §14.6.2 评价指标
# ═══════════════════════════════════════════


def cer(predicted: str, ground_truth: str) -> float:
    """字符错误率（CER）= 编辑距离 / 字符总数"""
    from Levenshtein import distance
    return distance(predicted, ground_truth) / max(len(ground_truth), 1)


def wer(predicted: str, ground_truth: str) -> float:
    """单词错误率（WER）"""
    from Levenshtein import distance
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    return distance(pred_words, gt_words) / max(len(gt_words), 1)


# ═══════════════════════════════════════════
# §14.6.7 TrOCR 微调完整代码
# ═══════════════════════════════════════════


class HandwritingDataset(Dataset):
    """手写数据集：CSV 格式 [image_path, text]"""

    def __init__(self, csv_path: str, img_root: str, processor,
                 max_target_length: int = 64):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"])
        text = str(row["text"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """批量计算 CER"""
    import Levenshtein
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        total_dist += Levenshtein.distance(pred, ref)
        total_len += max(len(ref), 1)
    return total_dist / total_len


def train():
    """TrOCR 微调主流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # 关键生成参数（不设置则训练/推理崩）
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.to(device)

    data_dir = Path(__file__).parent.parent / "data"
    train_set = HandwritingDataset(str(data_dir / "train.csv"), str(data_dir / "images"), processor)
    val_set = HandwritingDataset(str(data_dir / "val.csv"), str(data_dir / "images"), processor)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_cer = float("inf")
    for epoch in range(num_epochs):
        # --- 训练 ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # --- 验证 ---
        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                gen_ids = model.generate(pixel_values, max_length=64)
                pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)
                labels = batch["labels"]
                labels[labels == -100] = processor.tokenizer.pad_token_id
                ref_text = processor.batch_decode(labels, skip_special_tokens=True)
                preds.extend(pred_text)
                refs.extend(ref_text)

        val_cer = compute_cer(preds, refs)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, val_CER={val_cer * 100:.2f}%")

        if val_cer < best_cer:
            best_cer = val_cer
            model.save_pretrained("./trocr-finetuned-best")
            processor.save_pretrained("./trocr-finetuned-best")
            print(f"  -> 已保存（CER={val_cer * 100:.2f}%）")


# ═══════════════════════════════════════════
# NumPy 演示：自注意力 + 缩放因子对比
# ═══════════════════════════════════════════


def demo_self_attention_numpy():
    """纯 NumPy 实现 Self-Attention，展示内部计算细节"""
    def softmax_np(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    np.random.seed(42)
    L, d = 3, 4
    X = np.random.randn(L, d)
    W_q = np.random.randn(d, d) * 0.1
    W_k = np.random.randn(d, d) * 0.1
    W_v = np.random.randn(d, d) * 0.1

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    A = softmax_np(scores, axis=-1)
    Y = A @ V

    print("输入 X:\n", X.round(3))
    print("\n注意力权重 A:\n", A.round(3))
    print("行和验证:", A.sum(axis=1).round(6))
    print("\n输出 Y:\n", Y.round(3))


def demo_scale_comparison():
    """对比有无缩放时 softmax 的分布差异"""
    def softmax_np(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    np.random.seed(0)
    print("d_k      no_scale_max  scaled_max")
    for d_k in [16, 64, 256, 1024]:
        q = np.random.randn(d_k)
        K = np.random.randn(10, d_k)
        w_no = softmax_np(K @ q)
        w_yes = softmax_np(K @ q / np.sqrt(d_k))
        print(f"{d_k:4d}      {w_no.max():.3f}          {w_yes.max():.3f}")


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("§14.2 Self-Attention NumPy 演示")
    print("=" * 50)
    demo_self_attention_numpy()

    print("\n" + "=" * 50)
    print("§14.2.2 缩放因子对比实验")
    print("=" * 50)
    demo_scale_comparison()

    print("\n" + "=" * 50)
    print("§14.2.4 位置编码相似度")
    print("=" * 50)
    pe = positional_encoding(50, 128)
    print("PE shape:", pe.shape)
    for delta in [1, 5, 20]:
        sim = torch.nn.functional.cosine_similarity(
            pe[0].unsqueeze(0), pe[delta].unsqueeze(0)
        ).item()
        print(f"  PE(0) vs PE({delta:2d}) cos_sim = {sim:.4f}")

    print("\n" + "=" * 50)
    print("§14.3.2 PatchEmbedding 维度验证")
    print("=" * 50)
    patch = PatchEmbedding(384, 16, 768)
    dummy = torch.randn(2, 3, 384, 384)
    out = patch(dummy)
    print(f"输入: {dummy.shape} -> 输出: {out.shape}")
    print(f"patch 数量: {patch.num_patches}")

    print("\n" + "=" * 50)
    print("§14.3.3 因果掩码")
    print("=" * 50)
    mask = causal_mask(5)
    print(mask.int())

    print("\n§14.6.7 若要启动微调训练，请准备好 data/train.csv 和 data/val.csv（位于教材代码/data/ 目录）")
    print("  然后调用 train() 函数")
