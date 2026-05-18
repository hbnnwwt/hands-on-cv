"""校验教材中每章是否有'参考文献'节

用法：
    python tools/check_references.py [textbook_dir]
"""

from __future__ import annotations
import re
import sys
from pathlib import Path


def check_references(md_file: Path) -> dict:
    content = md_file.read_text(encoding='utf-8')
    has_section = bool(re.search(r'^##\s+参考文献', content, re.MULTILINE))
    items = re.findall(r'^\[(\d+)\]', content, re.MULTILINE)
    return {
        'file': md_file.name,
        'has_section': has_section,
        'item_count': len(items),
    }


def main():
    if len(sys.argv) > 1:
        textbook_dir = Path(sys.argv[1])
    else:
        textbook_dir = Path(__file__).parent.parent / "教材"

    if not textbook_dir.exists():
        print(f"错误：教材目录不存在: {textbook_dir}")
        sys.exit(1)

    print(f"校验参考文献: {textbook_dir.absolute()}\n")

    total = 0
    passed = 0
    failed = []

    for chapter_dir in sorted(textbook_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        if not (chapter_dir.name.startswith('第') or chapter_dir.name.startswith('附录')):
            continue

        md_files = [f for f in chapter_dir.glob("*.md")
                    if not f.name.startswith('章节') and not f.name.startswith('附录架构')]
        for md_file in md_files:
            total += 1
            result = check_references(md_file)
            if result['has_section'] and result['item_count'] >= 3:
                passed += 1
                print(f"  ✓ {chapter_dir.name}/{md_file.name}: {result['item_count']}条")
            else:
                status = "无参考文献节" if not result['has_section'] else f"仅{result['item_count']}条（需≥3）"
                print(f"  ✗ {chapter_dir.name}/{md_file.name}: {status}")
                failed.append(f"{chapter_dir.name}/{md_file.name}")

    print(f"\n=== 总计 ===")
    print(f"通过: {passed}/{total}")
    if failed:
        print(f"未通过 ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("全部通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
