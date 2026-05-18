"""核查教材中图表编号与正文引用的一致性

功能：
    1. 提取每章正文中的"图 X-Y"引用
    2. 检查引用出现顺序是否严格递增
    3. 检查编号是否连续（无跳跃）
    4. 报告错配和缺失

用法：
    python tools/check_figure_refs.py [textbook_dir]
"""

from __future__ import annotations
import re
import sys
from pathlib import Path


def extract_figure_refs(content: str) -> list[dict]:
    refs = []
    for m in re.finditer(r'图\s*(\d+)[-–—](\d+)', content):
        refs.append({
            'chapter': int(m.group(1)),
            'number': int(m.group(2)),
            'full': m.group(0),
            'pos': m.start(),
        })
    return refs


def check_order(refs: list[dict]) -> list[str]:
    issues = []
    prev_num = 0
    for ref in refs:
        if ref['number'] <= prev_num:
            issues.append(
                f"  - 引用顺序异常：先出现 图{ref['chapter']}-{prev_num}，"
                f"后出现 图{ref['chapter']}-{ref['number']}"
            )
        prev_num = ref['number']
    return issues


def check_continuity(refs: list[dict]) -> list[str]:
    issues = []
    if not refs:
        return issues
    numbers = sorted(set(r['number'] for r in refs))
    chapter = refs[0]['chapter']
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i-1] > 1:
            missing = range(numbers[i-1]+1, numbers[i])
            for n in missing:
                issues.append(f"  - 编号缺失：图{chapter}-{n} 未在正文中引用")
    return issues


def main():
    if len(sys.argv) > 1:
        textbook_dir = Path(sys.argv[1])
    else:
        textbook_dir = Path(__file__).parent.parent / "教材"

    if not textbook_dir.exists():
        print(f"错误：教材目录不存在: {textbook_dir}")
        sys.exit(1)

    print(f"图表编号一致性核查: {textbook_dir.absolute()}\n")

    total_issues = 0

    for chapter_dir in sorted(textbook_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        if not chapter_dir.name.startswith('第'):
            continue

        md_files = list(chapter_dir.glob("chapter*.md"))
        if not md_files:
            continue

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            refs = extract_figure_refs(content)

            if not refs:
                continue

            all_issues = []
            all_issues.extend(check_order(refs))
            all_issues.extend(check_continuity(refs))

            if all_issues:
                chapter_num = refs[0]['chapter']
                print(f"[第{chapter_num}章] 发现 {len(all_issues)} 个问题：")
                for issue in all_issues:
                    print(issue)
                total_issues += len(all_issues)
            else:
                chapter_num = refs[0]['chapter']
                print(f"  ✓ 第{chapter_num}章: {len(refs)}个图引用，顺序正确")

    print(f"\n=== 总计 ===")
    if total_issues > 0:
        print(f"共发现 {total_issues} 个问题")
        sys.exit(1)
    else:
        print("全部图引用顺序正确！")
        sys.exit(0)


if __name__ == "__main__":
    main()
