"""为教材生成全书术语索引

扫描所有章节的Markdown文件，提取以"**Term** — 翻译"格式定义的术语，
生成按字母排序的索引，包含每个术语出现的所有章节位置。
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict


def extract_terms_from_md(md_file: Path) -> dict:
    """从Markdown文件中提取术语

    匹配模式：**Term** — 中文翻译 或 **中文** — Term
    """
    content = md_file.read_text(encoding='utf-8')
    terms = defaultdict(list)

    # 模式1: **English Term** — 中文翻译
    pattern1 = re.compile(r'\*\*([A-Z][A-Za-z\s\-/]+)\*\*\s*[—-]\s*([一-鿿][^\n。.,]+)')
    for match in pattern1.finditer(content):
        en_term = match.group(1).strip()
        cn_term = match.group(2).strip()
        terms[en_term.lower()].append((en_term, cn_term))

    return dict(terms)


def find_term_locations(textbook_dir: Path, terms: dict) -> dict:
    """查找每个术语在各章节的出现位置"""
    locations = defaultdict(list)

    for md_file in textbook_dir.rglob("chapter*.md"):
        chapter_dir = md_file.parent.name
        chapter_num = chapter_dir.split('_')[0]   # 提取"第X章"
        content = md_file.read_text(encoding='utf-8').lower()

        for term_key in terms:
            if term_key.lower() in content:
                locations[term_key].append(chapter_num)

    return dict(locations)


def main():
    textbook_dir = Path(__file__).parent.parent / "教材"
    if not textbook_dir.exists():
        # 直接路径
        textbook_dir = Path("D:/downloads/计算机视觉微课/教材")

    print(f"扫描教材目录: {textbook_dir}")

    all_terms = {}
    for md_file in textbook_dir.rglob("*.md"):
        if "chapter" not in md_file.name.lower() and "appendix" not in md_file.name.lower():
            continue
        terms = extract_terms_from_md(md_file)
        for key, values in terms.items():
            all_terms.setdefault(key, []).extend(values)

    print(f"\n提取到 {len(all_terms)} 个术语")

    locations = find_term_locations(textbook_dir, all_terms)
    print(f"分析完出现位置")

    # 生成索引
    output = textbook_dir / "附录D_术语表" / "appendix_d_enhanced.md"

    with open(output, 'w', encoding='utf-8') as f:
        f.write("# 附录D 全书术语索引（增强版）\n\n")
        f.write("自动生成于扫描所有章节的术语定义。\n\n")

        # 按字母排序
        for key in sorted(all_terms.keys()):
            occurrences = all_terms[key]
            # 取第一次出现的英中文对照
            en, cn = occurrences[0]
            locs = locations.get(key, [])
            locs_str = ", ".join(sorted(set(locs)))
            f.write(f"**{en}** — {cn}\n")
            f.write(f"出现于：{locs_str}\n\n")

    print(f"\n已生成增强索引: {output}")


if __name__ == "__main__":
    main()
