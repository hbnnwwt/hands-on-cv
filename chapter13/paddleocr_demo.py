"""第13章：PaddleOCR 文字识别演示

教材对应：第13章 13.5 PaddleOCR实战

注意：运行前需要 pip install paddleocr paddlepaddle
"""

from __future__ import annotations
import sys
from pathlib import Path


def paddleocr_demo(image_path: str):
    """PaddleOCR 三行代码上手"""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("错误：未安装 paddleocr")
        print("请运行: pip install paddleocr paddlepaddle")
        return

    # 初始化OCR
    print("初始化 PaddleOCR ...")
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

    # 识别
    print(f"识别图像: {image_path}")
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        print("未检测到文字")
        return

    # 解析结果
    print("\n=== 识别结果 ===")
    for idx, line in enumerate(result[0]):
        bbox = line[0]      # 4个角点坐标
        text = line[1][0]   # 识别的文字
        score = line[1][1]  # 置信度

        # 计算外接矩形
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        print(f"#{idx + 1}: '{text}' (置信度: {score:.3f}) "
              f"位置: ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")


def visualize_ocr_result(image_path: str, output_path: str = "outputs/ocr_result.png"):
    """可视化OCR结果"""
    try:
        from paddleocr import PaddleOCR, draw_ocr
        from PIL import Image
    except ImportError:
        print("缺少依赖：pip install paddleocr paddlepaddle pillow")
        return

    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        print("未检测到文字")
        return

    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    image = Image.open(image_path).convert('RGB')
    annotated = draw_ocr(image, boxes, texts, scores)
    Image.fromarray(annotated).save(output_path)
    print(f"已保存可视化结果: {output_path}")


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        repo_root = Path(__file__).parent.parent
        image_path = str(repo_root / "data" / "answer_sheet" / "imgs" / "answer_sheet_1.png")

    Path("outputs").mkdir(exist_ok=True)

    paddleocr_demo(image_path)
    visualize_ocr_result(image_path)


if __name__ == "__main__":
    main()
