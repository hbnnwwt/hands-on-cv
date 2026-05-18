"""第12章：在MNIST上训练CNN

教材对应：第12章 12.7 PyTorch CNN实战

使用 LeNet 在 MNIST 上训练，演示完整的训练流程。
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import show_plot


class LeNet(nn.Module):
    """LeNet-5（针对32x32输入）"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # 下载数据到共享 data 目录
    data_dir = Path(__file__).parent.parent / "data"
    print("加载MNIST数据集...")
    train_data = datasets.MNIST(str(data_dir), train=True, download=True,
                                 transform=transform)
    test_data = datasets.MNIST(str(data_dir), train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # 模型
    model = LeNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("\n开始训练...")
    best_acc = 0
    history = {'train_loss': [], 'test_loss': [],
               'train_acc': [], 'test_acc': []}
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/5: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            Path("outputs").mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'outputs/best_lenet.pth')

    print(f"\n最佳测试准确率: {best_acc:.4f}")
    print("模型已保存到 outputs/best_lenet.pth")

    # 训练曲线可视化
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history['train_loss'], 'b-o', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-s', label='测试损失', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("LeNet 损失曲线")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-s', label='测试准确率', linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"LeNet 准确率曲线 (最佳测试={best_acc:.4f})")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()

    show_plot("outputs/lenet_training_curves.png")


if __name__ == "__main__":
    main()
