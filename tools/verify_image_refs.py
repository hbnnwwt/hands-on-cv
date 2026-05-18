"""校验教材中所有 ![](...) 图片引用是否有效

用法：
    python tools/verify_image_refs.py [textbook_dir]
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict


def find_image_refs(md_file: Path) -> list:
    """在Markdown文件中查找所有图片引用"""
    pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    refs = []
    for line_no, line in enumerate(md_file.read_text(encoding='utf-8').splitlines(), 1):
        for match in pattern.finditer(line):
            alt_text = match.group(1)
            image_path = match.group(2)
            refs.append({
                'line': line_no,
                'alt': alt_text,
                'path': image_path,
            })
    return refs


def verify_chapter(chapter_dir: Path) -> dict:
    """校验某一章的所有图片引用"""
    md_files = list(chapter_dir.glob("*.md"))
    if not md_files:
        return None

    chapter_name = chapter_dir.name
    result = {
        'chapter': chapter_name,
        'total': 0,
        'valid': 0,
        'invalid': [],
    }

    for md_file in md_files:
        refs = find_image_refs(md_file)
        for ref in refs:
            result['total'] += 1
            # 解析相对路径
            ref_path = (md_file.parent / ref['path']).resolve()

            if ref_path.exists():
                result['valid'] += 1
            else:
                # 检查 .png 替代版本
                if ref['path'].endswith('.svg'):
                    png_path = (md_file.parent / ref['path'].replace('.svg', '.png')).resolve()
                    if png_path.exists():
                        result['valid'] += 1
                        continue

                result['invalid'].append({
                    'file': md_file.name,
                    'line': ref['line'],
                    'alt': ref['alt'],
                    'path': ref['path'],
                })

    return result


def main():
    if len(sys.argv) > 1:
        textbook_dir = Path(sys.argv[1])
    else:
        textbook_dir = Path(__file__).parent.parent / "教材"

    if not textbook_dir.exists():
        print(f"错误：教材目录不存在: {textbook_dir}")
        sys.exit(1)

    print(f"校验教材目录: {textbook_dir.absolute()}\n")

    # 遍历所有章节子目录
    total_refs = 0
    total_valid = 0
    all_invalid = []

    for chapter_dir in sorted(textbook_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        if not (chapter_dir.name.startswith('第') or chapter_dir.name.startswith('附录')):
            continue

        result = verify_chapter(chapter_dir)
        if result is None:
            continue

        total_refs += result['total']
        total_valid += result['valid']
        all_invalid.extend(
            {**inv, 'chapter': result['chapter']}
            for inv in result['invalid']
        )

        status = "✓" if not result['invalid'] else f"✗ ({len(result['invalid'])}处失效)"
        print(f"  {status} {result['chapter']}: {result['valid']}/{result['total']}")

    print(f"\n=== 总计 ===")
    print(f"图片引用总数: {total_refs}")
    print(f"有效: {total_valid}")
    print(f"失效: {len(all_invalid)}")

    if all_invalid:
        print("\n失效的引用：")
        for inv in all_invalid:
            print(f"  - [{inv['chapter']}] {inv['file']}:{inv['line']}")
            print(f"    引用: {inv['path']}")


if __name__ == "__main__":
    main()
