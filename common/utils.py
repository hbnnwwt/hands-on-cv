"""共享工具函数：跨章节使用的通用辅助函数

主要功能：
- 图像读取与显示
- 文本/图像可视化辅助
- 性能计时
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """安全读取图像，处理常见错误

    参数：
        path: 图像路径
        grayscale: 是否以灰度模式读取

    返回：
        NumPy 图像数组

    异常：
        FileNotFoundError: 文件不存在
        ValueError: 文件无法解码为图像
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    # 统一使用 numpy 方式读取，避免 Windows 中文路径下 cv2.imread 报警告
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)

    if img is None:
        raise ValueError(f"无法解码图像: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    """安全保存图像，支持中文路径"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 兼容中文路径
    ext = path.suffix
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"图像编码失败: {path}")
    encoded.tofile(str(path))


@contextmanager
def timer(name: str = "操作"):
    """上下文管理器：测量代码块耗时

    用法:
        with timer("图像处理"):
            result = process(image)
        # 输出: [图像处理] 耗时: 123.4 ms
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] 耗时: {elapsed * 1000:.1f} ms")


def show_images(*images: np.ndarray,
                titles: Optional[list[str]] = None,
                window_size: Tuple[int, int] = (10, 6)) -> None:
    """用 matplotlib 显示多张图像（保存并用 cv2.imshow 弹窗）

    参数:
        *images: 一张或多张图像
        titles: 对应的标题列表
        window_size: 窗口大小
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    n = len(images)
    if titles is None:
        titles = [f"图像 {i + 1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=window_size)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    show_plot()


def show_plot(save_path: str = "outputs/plot_result.png") -> None:
    """保存 matplotlib 图表并用 cv2.imshow 弹窗显示

    替代 plt.show()，兼容无 GUI 后端的环境。
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f"图表已保存: {save_path}")

    # 用 cv2.imshow 弹窗显示
    plot_img = read_image(save_path)
    cv2.imshow("plot", plot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，返回 Path 对象"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_answer_gt(gt_path: str | Path) -> dict[str, str]:
    """从 GT 文件加载答题卡标准答案

    GT 文件格式：每行 "题号\\t答案"，"-" 表示未填涂

    参数：
        gt_path: GT 文本文件路径（如 data/answer_sheet/gt/answer_sheet_1.txt）

    返回：
        {"1": "A", "2": "B", ...} 字典，"-" 表示该题未填涂
    """
    answers = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                answers[parts[0]] = parts[1]
    return answers


def find_gt_for_image(image_path: str | Path) -> Path | None:
    """根据图片路径自动查找对应的 GT 文件

    data/answer_sheet/imgs/test_set_01.png → data/answer_sheet/gt/test_set_01.txt

    返回：
        GT 文件路径，找不到则返回 None
    """
    image_path = Path(image_path)
    gt_path = image_path.parent.parent / "gt" / (image_path.stem + ".txt")
    return gt_path if gt_path.exists() else None


def load_student_id_from_gt(gt_path: str | Path) -> Optional[str]:
    """从 GT 文件第一行读取学号

    GT 文件单数页（第1/3/5页）首行格式："id <学号>"，
    若未指定学号则为 "id -"。偶数页通常没有 id 行。

    返回：
        学号字符串；若首行不是 id 行或学号为 "-"，返回 None。
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        return None
    with open(gt_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    if not first_line.lower().startswith('id'):
        return None
    parts = first_line.split()
    if len(parts) < 2:
        return None
    student_id = parts[1].strip()
    if not student_id or student_id == '-':
        return None
    return student_id


def list_student_ids_from_segmented(base_dir: str | Path | None = None) -> list[str]:
    """从 segmented 目录扫描所有学号

    文件命名: {学号}_{kind}.png，提取唯一学号列表并排序。
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data" / "answer_sheet" / "segmented"
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    ids = set()
    for p in base_dir.glob("*_choice.png"):
        ids.add(p.stem.rsplit("_", 1)[0])
    return sorted(ids)


def find_gt_for_student_id(student_id: str,
                           gt_dir: str | Path | None = None) -> list[Path]:
    """根据学号反查 GT 文件

    扫描 gt_dir 下所有 .txt，找到首行 id 匹配的 GT 文件，
    并自动带上配对的偶数页 GT（如 answer_sheet_1 → answer_sheet_2）。
    """
    if gt_dir is None:
        gt_dir = Path(__file__).parent.parent / "data" / "answer_sheet" / "gt"
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return []
    matched = []
    for gt_file in sorted(gt_dir.glob("*.txt")):
        sid = load_student_id_from_gt(gt_file)
        if sid == student_id:
            matched.append(gt_file)
            # 查找配对的偶数页 GT
            stem = gt_file.stem
            for i in range(len(stem) - 1, -1, -1):
                if stem[i].isdigit():
                    digit = int(stem[i])
                    if digit % 2 == 1:
                        paired_stem = stem[:i] + str(digit + 1) + stem[i + 1:]
                        paired = gt_dir / (paired_stem + gt_file.suffix)
                        if paired.exists():
                            matched.append(paired)
                    break
    return matched


def find_segmented_image_for_id(student_id: str, kind: str,
                                base_dir: str | Path | None = None) -> Path | None:
    """根据学号在 segmented 目录查找区域分割图

    参数：
        student_id: 学号（GT 首行解析得到）
        kind: 'choice' | 'judge' | 'student_id' | 'essay'
        base_dir: 分割目录，默认 repo_root/data/answer_sheet/segmented

    返回：
        分割图 Path，未找到则 None
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data" / "answer_sheet" / "segmented"
    base_dir = Path(base_dir)
    seg_path = base_dir / f"{student_id}_{kind}.png"
    return seg_path if seg_path.exists() else None
