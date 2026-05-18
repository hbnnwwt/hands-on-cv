"""第1章 1.6 节：第一段OpenCV代码 — 读取并显示图像

教材对应：第1章 1.6.3 OpenCV "Hello World"
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# 添加项目根目录到路径，以导入 common.utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import read_image, show_plot


def hello_opencv(image_path: str) -> None:
    """读取图像并显示（OpenCV版本）"""
    try:
        img = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取图像 {image_path}: {e}")
        return

    print(f"图像形状: {img.shape}")
    print(f"数据类型: {img.dtype}")
    print(f"像素范围: [{img.min()}, {img.max()}]")

    cv2.imshow("我的第一张图像", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hello_matplotlib(image_path: str) -> None:
    """读取图像并显示（Matplotlib版本，跨平台更稳定）"""
    try:
        img = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取图像 {image_path}: {e}")
        return

    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # OpenCV使用BGR，Matplotlib使用RGB，需要转换
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title("我的第一张图像")
    plt.axis("off")
    plt.tight_layout()
    show_plot("outputs/hello_matplotlib.png")


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认使用仓库的测试图像
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_1.png")
        if not Path(image_path).exists():
            print("提示：测试图像不存在。先运行：")
            print("  python common/test_images.py")
            sys.exit(1)

    print(f"读取图像：{image_path}\n")
    hello_matplotlib(image_path)


if __name__ == "__main__":
    main()
