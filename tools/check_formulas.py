"""扫描并修复教材中公式的编号

功能：
    1. 扫描模式：报告所有缺少 \\tag{X.Y} 编号的公式
    2. 修复模式：自动为未编号公式添加 \\tag{X.Y}

用法：
    python tools/check_formulas.py [textbook_dir]           # 扫描模式
    python tools/check_formulas.py [textbook_dir] --fix     # 修复模式
"""

from __future__ import annotations
import re
import sys
import shutil
from pathlib import Path


def extract_chapter_num(chapter_dir: Path) -> int | None:
    m = re.match(r'第(\d+)章', chapter_dir.name)
    if m:
        return int(m.group(1))
    appendix_map = {'附录A': 101, '附录B': 102, '附录C': 103, '附录D': 104}
    for prefix, num in appendix_map.items():
        if chapter_dir.name.startswith(prefix):
            return num
    return None


def scan_formulas(content: str, chapter_num: int) -> list[dict]:
    results = []
    counter = 1
    for m in re.finditer(r'\$\$\n?(.*?)\$\$', content, re.DOTALL):
        formula = m.group(1).strip()
        has_tag = bool(re.search(r'\\tag\{', formula))
        is_special = bool(re.search(r'\\begin\{(matrix|cases|aligned|array|pmatrix|bmatrix)', formula))
        results.append({
            'start': m.start(),
            'formula_preview': formula[:80],
            'has_tag': has_tag,
            'is_special': is_special,
            'expected_tag': f'{chapter_num}.{counter}' if not has_tag else None,
        })
        if not has_tag:
            counter += 1
        else:
            tag_match = re.search(r'\\tag\{(\d+)\.(\d+)\}', formula)
            if tag_match:
                counter = int(tag_match.group(2)) + 1
    return results


def fix_formulas(content: str, chapter_num: int) -> str:
    counter = 1

    def replace_match(m):
        nonlocal counter
        formula = m.group(1).strip()
        if re.search(r'\\tag\{', formula):
            tag_match = re.search(r'\\tag\{(\d+)\.(\d+)\}', formula)
            if tag_match:
                counter = int(tag_match.group(2)) + 1
            return m.group(0)

        is_special = bool(re.search(r'\\begin\{(matrix|cases|aligned|array|pmatrix|bmatrix)', formula))
        if is_special:
            counter += 1
            return m.group(0)

        tag = f'{chapter_num}.{counter}'
        counter += 1
        return f'$$\n{formula}\n\\tag{{{tag}}}\n$$'

    return re.sub(r'\$\$\n?(.*?)\$\$', replace_match, content, flags=re.DOTALL)


def main():
    fix_mode = '--fix' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--fix']

    if args:
        textbook_dir = Path(args[0])
    else:
        textbook_dir = Path(__file__).parent.parent / "教材"

    if not textbook_dir.exists():
        print(f"错误：教材目录不存在: {textbook_dir}")
        sys.exit(1)

    mode_str = "修复模式" if fix_mode else "扫描模式"
    print(f"公式编号{mode_str}: {textbook_dir.absolute()}\n")

    total_formulas = 0
    untagged = 0
    files_with_issues = []

    for chapter_dir in sorted(textbook_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        if not (chapter_dir.name.startswith('第') or chapter_dir.name.startswith('附录')):
            continue

        chapter_num = extract_chapter_num(chapter_dir)
        if chapter_num is None:
            continue

        md_files = [f for f in chapter_dir.glob("*.md")
                    if not f.name.startswith('章节') and not f.name.startswith('附录架构')]

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            results = scan_formulas(content, chapter_num)
            file_untagged = [r for r in results if not r['has_tag'] and not r['is_special']]

            total_formulas += len(results)
            untagged += len(file_untagged)

            if file_untagged:
                files_with_issues.append(md_file)
                print(f"  ✗ {chapter_dir.name}/{md_file.name}: {len(file_untagged)}/{len(results)} 未编号")
                for r in file_untagged:
                    print(f"      → 预期 \\tag{{{r['expected_tag']}}}: {r['formula_preview']}...")

            if fix_mode and file_untagged:
                backup = md_file.with_suffix('.md.bak')
                shutil.copy2(md_file, backup)
                new_content = fix_formulas(content, chapter_num)
                md_file.write_text(new_content, encoding='utf-8')
                print(f"      ✓ 已修复并备份至 {backup.name}")

    print(f"\n=== 总计 ===")
    print(f"公式总数: {total_formulas}")
    print(f"未编号: {untagged}")

    if fix_mode:
        print(f"已修复文件: {len(files_with_issues)}")
    else:
        if untagged > 0:
            print(f"\n运行 python tools/check_formulas.py --fix 以自动修复")
            sys.exit(1)
        else:
            print("全部公式已编号！")
            sys.exit(0)


if __name__ == "__main__":
    main()
