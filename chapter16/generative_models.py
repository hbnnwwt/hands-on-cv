"""第16章：生成式模型与视觉大模型综合代码

教材对应：第16章 16.2 / 16.4 / 16.5 / 16.6

包含：
  - GAN（Generator / Discriminator / 训练循环）
  - Stable Diffusion 文本生成图像推理
  - CLIP 零样本分类 / ZeroShotClassifier
  - CLIP 答题卡版式分类
  - CLIP InfoNCE 对比损失
  - SAM 分割示例
  - DINOv2 特征提取
  - 多模态大模型 API 调用示例
"""

from __future__ import annotations
from pathlib import Path

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os


# ============================================================
#  §16.2 / §16.8.2  GAN —— 完整 MNIST 实现
# ============================================================

class Generator(nn.Module):
    """GAN 生成器：从随机噪声 z 生成 28x28 灰度图。"""

    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28), nn.Tanh(),  # 输出 [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """GAN 判别器：判断图像是真实的还是生成的。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gan(G: nn.Module, D: nn.Module, dataloader: DataLoader,
              epochs: int = 30, device: str = "cpu") -> None:
    """GAN 对抗训练循环（§16.2.3 训练过程）。

    关键：Adam 的 beta1=0.5（DCGAN 论文经验），而非默认 0.9。
    """
    G.to(device)
    D.to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    os.makedirs("samples", exist_ok=True)
    fixed_z = torch.randn(64, 100, device=device)

    for epoch in range(epochs):
        d_losses, g_losses = [], []
        for real, _ in dataloader:
            real = real.to(device)
            bs = real.size(0)
            ones = torch.ones(bs, 1, device=device)
            zeros = torch.zeros(bs, 1, device=device)

            # ---- 训练 D ----
            z = torch.randn(bs, 100, device=device)
            fake = G(z)
            d_loss = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # ---- 训练 G ----
            g_loss = criterion(D(fake), ones)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        print(f"Epoch {epoch + 1:02d}  D_loss={avg_d:.4f}  G_loss={avg_g:.4f}")

        # 保存生成样本
        with torch.no_grad():
            samples = G(fixed_z).cpu()
            save_image(samples * 0.5 + 0.5,
                       f"samples/epoch_{epoch + 1:02d}.png", nrow=8)


# ============================================================
#  §16.3.4 / §16.6.4  Stable Diffusion 推理
# ============================================================

def stable_diffusion_generate(prompt: str, output_path: str = "output.png",
                              model_name: str = "runwayml/stable-diffusion-v1-5",
                              num_steps: int = 25) -> None:
    """使用 Stable Diffusion 从文本生成图像。

    需要: pip install diffusers transformers accelerate
    显存需求: 约 4 GB (float16)
    """
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    image = pipe(prompt, num_inference_steps=num_steps).images[0]
    image.save(output_path)
    print(f"图像已保存到 {output_path}")


def stable_diffusion_data_augmentation(prompts: list[str],
                                       output_dir: str = "synthetic") -> None:
    """用 Stable Diffusion 批量生成训练数据（§16.6.4 数据增强）。"""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to("cuda")

    os.makedirs(output_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=25).images[0]
        image.save(os.path.join(output_dir, f"{i:04d}.png"))
    print(f"已生成 {len(prompts)} 张合成图像到 {output_dir}/")


# ============================================================
#  §16.4.2 / §16.4.4  CLIP 零样本分类
# ============================================================

def clip_zero_shot_demo(image_path: str = "photo.jpg") -> None:
    """CLIP 零样本分类演示（§16.4.2）。

    需要: pip install clip (openai/CLIP)
    """
    import clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    class_descriptions = [
        "一张猫的照片",
        "一张狗的照片",
        "一张鸟的照片",
        "一张汽车的照片",
    ]
    text = clip.tokenize(class_descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # L2 归一化后求内积（余弦相似度）
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    for desc, prob in zip(class_descriptions, similarities[0]):
        print(f"{desc}: {prob.item():.2%}")


class ZeroShotClassifier:
    """基于 CLIP 的零样本分类器（§16.4.4）。"""

    def __init__(self, model_name: str = "ViT-B/32"):
        import clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def classify(self, image_path: str, class_names: List[str],
                 template: str = "一张{}的照片") -> dict:
        """对单张图像做零样本分类。

        参数:
            image_path:  图像路径
            class_names: 类别名列表（中文/英文均可）
            template:    描述模板
        返回:
            {class_name: probability}
        """
        from PIL import Image

        image = self.preprocess(
            Image.open(image_path)
        ).unsqueeze(0).to(self.device)

        descriptions = [template.format(name) for name in class_names]

        import clip
        text = clip.tokenize(descriptions).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return dict(zip(class_names, similarities[0].cpu().numpy().tolist()))


# ============================================================
#  §16.4.5  CLIP 答题卡版式分类（HuggingFace 版）
# ============================================================

def clip_answer_sheet_classify(image_path: str = "answer_sheet.jpg") -> None:
    """使用 HuggingFace CLIP 对答题卡版式做零样本分类（§16.4.5）。

    需要: pip install transformers
    """
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    class_names = ["A4单面答题卡", "A3双面答题卡", "A4试卷版", "作业纸"]
    text_prompts = [f"a photo of a {name}" for name in class_names]

    image = Image.open(image_path)

    inputs = processor(text=text_prompts, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    for name, prob in zip(class_names, probs[0]):
        print(f"{name}: {prob.item():.2%}")


# ============================================================
#  §16.8.6  CLIP InfoNCE 对比损失
# ============================================================

def clip_loss(image_features: torch.Tensor,
              text_features: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    """CLIP 对比学习损失 —— InfoNCE（§16.8.6）。

    核心：配对的图文相似度高、错配的低。
    labels = 对角线索引（第 i 个图像与第 i 个文本配对）。
    """
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = logit_scale * image_features @ text_features.T  # (N, N)
    labels = torch.arange(len(logits), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)       # 图像→文本
    loss_t2i = F.cross_entropy(logits.T, labels)     # 文本→图像
    return (loss_i2t + loss_t2i) / 2


# ============================================================
#  §16.5.2  SAM 分割示例
# ============================================================

def sam_segment_example(image_path: str = "answer_sheet.jpg",
                        checkpoint: str = "sam_vit_h_4b8939.pth",
                        point: tuple[int, int] = (500, 375)) -> None:
    """使用 SAM 进行点提示分割（§16.5.2）。

    需要: pip install segment-anything
    """
    import numpy as np
    from PIL import Image
    from segment_anything import sam_model_registry, SamPredictor

    image = np.array(Image.open(image_path).convert("RGB"))

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    print(f"分割掩码形状: {masks.shape}, 置信度: {scores}")


# ============================================================
#  §16.5.3  DINOv2 特征提取
# ============================================================

def dinov2_extract_features(image_path: str = "photo.jpg") -> torch.Tensor:
    """使用 DINOv2 提取图像特征（§16.5.3）。

    需要: pip install torch torchvision
    """
    from torchvision import transforms as T
    from PIL import Image

    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    features = dinov2(image)  # (1, 768)
    print(f"特征维度: {features.shape}")
    return features


# ============================================================
#  §16.5.4  多模态大模型 API 调用示例
# ============================================================

def vlm_analyze_answer_sheet(image_path: str = "answer_sheet.jpg") -> str:
    """调用 Claude 视觉 API 分析答题卡（§16.5.4）。

    需要: pip install anthropic
    """
    import anthropic
    import base64

    client = anthropic.Anthropic()

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "这是一份学生答题卡。请识别学生选择了哪些选项，"
                            "并给出每道题的答案。",
                },
            ],
        }],
    )

    return message.content[0].text


# ============================================================
#  运行入口：GAN 训练示例
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ---------- 准备 MNIST 数据 ----------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
    ])
    data_dir = Path(__file__).parent.parent / "data"
    dataset = datasets.MNIST(root=str(data_dir), train=True,
                             transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    # ---------- 训练 GAN ----------
    print("\n=== GAN 训练 ===")
    G = Generator()
    D = Discriminator()
    train_gan(G, D, loader, epochs=30, device=device)

    print("\n训练完成！生成样本保存在 samples/ 目录下。")
