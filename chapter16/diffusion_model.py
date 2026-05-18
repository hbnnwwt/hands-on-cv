"""第16章：扩散模型（Diffusion Model）核心代码

教材对应：第16章 16.3 / 16.8.3 / 16.8.4 / 16.9.3

包含：
  - 噪声调度（线性 / cosine）
  - 前向加噪（DDPM 闭式解）
  - 简化训练循环
  - 前向加噪过程可视化
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import show_plot


# ============================================================
#  §16.3.2  噪声调度（Noise Schedule）
# ============================================================

def linear_beta_schedule(T: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """线性噪声调度，DDPM 原始方案。"""
    return torch.linspace(beta_start, beta_end, T)


# §16.9.3  cosine schedule —— 信号衰减更平缓，FID 更低
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM 提出的 cosine 噪声调度。"""
    steps = torch.arange(T + 1) / T
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0, 0.999)


class GaussianDiffusion:
    """封装 DDPM 前向加噪与相关常量。

    核心公式（§16.8.3 闭式解）：
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    """

    def __init__(self, T: int = 1000, schedule: str = "linear"):
        if schedule == "linear":
            betas = linear_beta_schedule(T)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"未知 schedule: {schedule}")

        self.T = T
        # 注册为 buffer，随模型 .to(device) 移动
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """前向加噪：从 x_0 一步采样 x_t（§16.8.3 闭式解）。

        参数:
            x_0: 原始图像 (B, C, H, W)
            t:   时间步索引 (B,)，取值 [0, T)
            noise: 可选的预采样噪声，默认随机
        返回:
            x_t: 加噪后的图像
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 将 alpha_bar[t] reshape 为 (B, 1, 1, 1) 以便广播
        a_bar = self.alpha_bar.to(x_0.device)[t]
        shape = (-1,) + (1,) * (x_0.ndim - 1)
        sqrt_a_bar = a_bar.sqrt().view(shape)
        sqrt_one_minus_a_bar = (1 - a_bar).sqrt().view(shape)

        return sqrt_a_bar * x_0 + sqrt_one_minus_a_bar * noise


# ============================================================
#  §16.3.3  简化训练循环
# ============================================================

def train_diffusion(model: nn.Module, dataloader: DataLoader,
                    T: int = 1000, epochs: int = 50,
                    lr: float = 1e-4, device: str = "cpu") -> None:
    """简化的 Diffusion 训练循环。

    训练目标：让网络 eps_theta(x_t, t) 预测真实噪声 eps。
    损失 = MSE(eps_theta, eps)（§16.8.3 简化损失）
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    diffusion = GaussianDiffusion(T, schedule="cosine")

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for x_0, _ in dataloader:
            x_0 = x_0.to(device)
            optimizer.zero_grad()

            # 1. 随机采样时间步
            t = torch.randint(0, T, (x_0.size(0),), device=device)

            # 2. 采样噪声
            noise = torch.randn_like(x_0)

            # 3. 前向加噪得到 x_t
            x_t = diffusion.q_sample(x_0, t, noise)

            # 4. 网络预测噪声
            predicted_noise = model(x_t, t)

            # 5. MSE 损失
            loss = ((predicted_noise - noise) ** 2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_0.size(0)
            count += x_0.size(0)

        avg_loss = total_loss / count
        print(f"Epoch {epoch + 1:03d}  loss={avg_loss:.6f}")


# ============================================================
#  §16.8.4  前向加噪过程可视化
# ============================================================

def visualize_forward_process(save_path: str = "diffusion_forward.png",
                              T: int = 1000) -> None:
    """将一张 MNIST 数字逐步加噪并保存可视化。

    直观感受 alpha_bar_t 如何把信号淹没在噪声中：
      t=0 → 原图清晰；t=999 → 完全是噪声
    """
    diffusion = GaussianDiffusion(T, schedule="linear")

    data_dir = Path(__file__).parent.parent / "data"
    ds = datasets.MNIST(str(data_dir), train=True, download=True,
                        transform=transforms.ToTensor())
    x0 = ds[0][0] * 2 - 1  # 归一化到 [-1, 1]

    ts = [0, 50, 100, 200, 500, 999]
    fig, axes = plt.subplots(1, len(ts), figsize=(14, 2.5))
    torch.manual_seed(42)
    eps = torch.randn_like(x0)

    for ax, t in zip(axes, ts):
        t_tensor = torch.tensor([t])
        x_t = diffusion.q_sample(x0.unsqueeze(0), t_tensor, eps.unsqueeze(0))
        a_bar = diffusion.alpha_bar[t]
        ax.imshow(x_t.squeeze() * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t}\nsqrt_alpha_bar={a_bar.sqrt():.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"前向加噪可视化已保存到 {save_path}")
    show_plot()


# ============================================================
#  运行入口
# ============================================================

if __name__ == "__main__":
    print("=== 前向加噪可视化 ===")
    visualize_forward_process()
