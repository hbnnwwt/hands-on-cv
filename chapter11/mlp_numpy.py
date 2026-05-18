"""第11章：MLP的NumPy手动实现

教材对应：第11章 11.6 反向传播算法

完整的两层MLP，从前向传播到反向传播全部用NumPy手写。
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import show_plot


class MLPNumpy:
    """两层多层感知机的NumPy实现"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.01):
        # Xavier初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.lr = learning_rate

        # 缓存中间变量（用于反向传播）
        self.cache = {}

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def softmax(x):
        # 数值稳定的softmax
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播

        参数：
            x: shape (batch_size, input_dim)
        返回：
            概率分布 shape (batch_size, output_dim)
        """
        # 第一层
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)

        # 第二层
        z2 = a1 @ self.W2 + self.b2
        out = self.softmax(z2)

        # 缓存
        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['out'] = out

        return out

    def backward(self, y_true: np.ndarray) -> dict:
        """反向传播

        参数：
            y_true: one-hot标签 shape (batch_size, output_dim)
        返回：
            梯度字典
        """
        batch_size = y_true.shape[0]
        out = self.cache['out']
        a1 = self.cache['a1']
        z1 = self.cache['z1']
        x = self.cache['x']

        # 输出层梯度（softmax + cross-entropy的简化形式）
        dz2 = (out - y_true) / batch_size
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 隐藏层梯度
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_grad(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update(self, grads: dict) -> None:
        """SGD更新"""
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']

    def compute_loss(self, y_true: np.ndarray) -> float:
        """交叉熵损失"""
        out = self.cache['out']
        # 防止log(0)
        out_clipped = np.clip(out, 1e-12, 1.0 - 1e-12)
        return -np.sum(y_true * np.log(out_clipped)) / y_true.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = True) -> list:
        """训练"""
        n_samples = X.shape[0]
        n_classes = self.W2.shape[1]

        # 转one-hot
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1

        history = []
        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            total_loss = 0
            correct = 0

            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                X_batch = X[idx]
                y_batch = y_onehot[idx]

                # 前向
                out = self.forward(X_batch)
                loss = self.compute_loss(y_batch)
                total_loss += loss * len(idx)

                # 反向
                grads = self.backward(y_batch)
                self.update(grads)

                # 准确率
                preds = np.argmax(out, axis=1)
                correct += np.sum(preds == y[idx])

            avg_loss = total_loss / n_samples
            accuracy = correct / n_samples
            history.append({'epoch': epoch, 'loss': avg_loss, 'acc': accuracy})

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return np.argmax(self.forward(X), axis=1)


def demo():
    """在小型分类数据上演示"""
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    # 生成数据
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练
    print("=== MLP训练 ===")
    model = MLPNumpy(input_dim=2, hidden_dim=16, output_dim=2,
                     learning_rate=0.05)
    history = model.fit(X_train, y_train, epochs=100, verbose=True)

    # 评估
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")

    # 训练曲线可视化
    epochs = [h['epoch'] + 1 for h in history]
    losses = [h['loss'] for h in history]
    accs = [h['acc'] for h in history]

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, losses, 'b-', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练损失曲线")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, accs, 'g-', linewidth=2)
    ax2.axhline(y=accuracy, color='r', linestyle='--', alpha=0.6,
                label=f'测试准确率={accuracy:.3f}')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("训练准确率曲线")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()

    Path("outputs").mkdir(exist_ok=True)
    show_plot("outputs/mlp_training_curves.png")


if __name__ == "__main__":
    np.random.seed(42)
    demo()
