"""校验教材中每章是否有'本章学习目标'块

用法：
    python tools/check_learning_goals.py [textbook_dir]
"""

from __future__ import annotations
import io
import re
import sys
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def check_learning_goals(md_file: Path) -> dict:
    """检查单个md文件是否包含学习目标块"""
    content = md_file.read_text(encoding='utf-8')
    has_block = bool(re.search(r'>\s*\*\*本章学习目标\*\*', content))
    items = re.findall(r'>\s*\d+\.\s*\*\*\[', content)
    return {
        'file': md_file.name,
        'has_block': has_block,
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

    print(f"校验学习目标: {textbook_dir.absolute()}\n")

    total = 0
    passed = 0
    failed = []

    for chapter_dir in sorted(textbook_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        if not (chapter_dir.name.startswith('第') or chapter_dir.name.startswith('附录')):
            continue

        md_files = list(chapter_dir.glob("*.md"))
        content_files = [f for f in md_files if not f.name.startswith('章节') and not f.name.startswith('附录架构')]
        if not content_files:
            continue

        for md_file in content_files:
            total += 1
            result = check_learning_goals(md_file)
            min_items = 2 if chapter_dir.name.startswith('附录') else 3
            if result['has_block'] and result['item_count'] >= min_items:
                passed += 1
                print(f"  ✓ {chapter_dir.name}/{md_file.name}: {result['item_count']}条目标")
            else:
                status = "无学习目标块" if not result['has_block'] else f"仅{result['item_count']}条（需≥{min_items}）"
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
