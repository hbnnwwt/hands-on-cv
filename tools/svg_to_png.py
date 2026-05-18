"""SVG → PNG 批量转换工具

将教材所有章节的 SVG 插图转换为 PNG，方便：
1. 不支持 SVG 的Markdown预览器
2. 印刷出版（PDF嵌入PNG更稳定）
3. 双击预览

用法：
    python tools/svg_to_png.py

依赖（任选其一）：
    pip install cairosvg
    # 或
    pip install svglib reportlab
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Optional


def get_converter() -> Optional[Callable]:
    """返回可用的SVG→PNG转换函数"""

    # 方法1：cairosvg（推荐，输出质量最好）
    try:
        import cairosvg

        def convert_cairo(svg_path: Path, png_path: Path, dpi: int = 150):
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                output_width=1600,  # 高分辨率
            )
        print("✓ 使用 cairosvg 转换器")
        return convert_cairo
    except ImportError:
        pass

    # 方法2：svglib + reportlab（纯Python，无系统依赖）
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM

        def convert_svglib(svg_path: Path, png_path: Path, dpi: int = 150):
            drawing = svg2rlg(str(svg_path))
            renderPM.drawToFile(drawing, str(png_path), fmt="PNG", dpi=dpi)
        print("✓ 使用 svglib 转换器")
        return convert_svglib
    except ImportError:
        pass

    # 方法3：通过 inkscape 命令行（如果系统已安装）
    import shutil
    if shutil.which('inkscape'):
        import subprocess

        def convert_inkscape(svg_path: Path, png_path: Path, dpi: int = 150):
            subprocess.run([
                'inkscape',
                '--export-type=png',
                f'--export-dpi={dpi}',
                f'--export-filename={png_path}',
                str(svg_path),
            ], check=True, capture_output=True)
        print("✓ 使用 inkscape 命令行转换器")
        return convert_inkscape

    print("\n错误：没有可用的SVG转换器。请安装以下任一：")
    print("  pip install cairosvg  # 推荐")
    print("  pip install svglib reportlab")
    print("  或安装 Inkscape: https://inkscape.org/")
    return None


def convert_all_svgs(textbook_dir: Path, converter: Callable) -> dict:
    """转换教材目录下的所有SVG"""
    stats = {'total': 0, 'success': 0, 'failed': 0, 'failures': []}

    svg_files = list(textbook_dir.rglob("*.svg"))
    stats['total'] = len(svg_files)

    print(f"\n找到 {len(svg_files)} 个SVG文件\n")

    for svg_path in svg_files:
        png_path = svg_path.with_suffix('.png')

        try:
            converter(svg_path, png_path)
            stats['success'] += 1
            print(f"  ✓ {svg_path.relative_to(textbook_dir)}")
        except Exception as e:
            stats['failed'] += 1
            stats['failures'].append((str(svg_path), str(e)))
            print(f"  ✗ {svg_path.relative_to(textbook_dir)}: {e}")

    return stats


def main():
    if len(sys.argv) > 1:
        textbook_dir = Path(sys.argv[1])
    else:
        # 默认是脚本所在目录的父目录
        textbook_dir = Path(__file__).parent.parent / "教材"

    if not textbook_dir.exists():
        print(f"错误：教材目录不存在: {textbook_dir}")
        sys.exit(1)

    print(f"教材目录: {textbook_dir.absolute()}")

    converter = get_converter()
    if converter is None:
        sys.exit(1)

    stats = convert_all_svgs(textbook_dir, converter)

    print("\n=== 转换完成 ===")
    print(f"总数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")

    if stats['failures']:
        print("\n失败的文件：")
        for path, err in stats['failures']:
            print(f"  - {path}: {err}")


if __name__ == "__main__":
    main()
