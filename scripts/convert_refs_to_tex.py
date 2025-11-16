import re
import os

IN_PATH = 'output/ref_list_2023_2025_gbt7714.lark.md'
OUT_PATH = 'output/ref_list_2023_2025_gbt7714.tex'

# Escape LaTeX text (non-URL) special chars
TEXT_ESCAPES = [
    ('&', '\\&'),
    ('%', '\\%'),
    ('#', '\\#'),
    ('_', '\\_'),
    ('~', '\\textasciitilde{}'),
    ('^', '\\textasciicircum{}'),
]

# Escape URL argument special chars for \href{...}{...}
URL_ESCAPES = [
    ('&', '\\&'),
    ('%', '\\%'),
    ('#', '\\#'),
    ('_', '\\_'),
    ('~', '\\~{}'),
    ('^', '\\^{}'),
]

def escape_text(s: str) -> str:
    for old, new in TEXT_ESCAPES:
        s = s.replace(old, new)
    return s

def escape_url(s: str) -> str:
    for old, new in URL_ESCAPES:
        s = s.replace(old, new)
    return s

# Detect type bracket like ［J］, ［C/EB/OL］ etc.
TYPE_RE = re.compile(r'［([A-Z]+(?:/[A-Z]+)*)］')

# Extract first URL after a given marker
def replace_links_with_href(line: str) -> str:
    # 先处理“开放获取PDF链接：URL”，避免与下一条的“链接：URL”重复替换
    line = re.sub(r'(开放获取PDF链接)：\s*([^\s）]+)', lambda m: f"{m.group(1)}：\\href{{{escape_url(m.group(2))}}}{{{escape_url(m.group(2))}}}", line)
    # 再处理一般“链接：URL”，排除前缀为“开放获取PDF”的情况
    line = re.sub(r'(?<!开放获取PDF)(链接)：\s*([^\s）]+)', lambda m: f"{m.group(1)}：\\href{{{escape_url(m.group(2))}}}{{{escape_url(m.group(2))}}}", line)
    return line

# Append DOI hyperlink after DOI文本
DOI_RE = re.compile(r'(DOI)：\s*([\w./-]+)')

def append_doi_href(line: str) -> str:
    def _add(m):
        doi = m.group(2)
        url = 'https://doi.org/' + doi
        return f"{m.group(1)}：{doi}（\\href{{{escape_url(url)}}}{{{escape_url(url)}}}）"
    return DOI_RE.sub(_add, line)

# Remove the line-number prefix if present in file (shouldn't be in raw file)
LINE_PREFIX_RE = re.compile(r'^【\d+】')

# Extract leading index ［n］
INDEX_RE = re.compile(r'^\s*［(\d+)］')

# Build one \bibitem line

def build_bibitem(raw_line: str) -> str:
    line = raw_line.strip()
    # Remove read-tool line number prefix if any
    line = LINE_PREFIX_RE.sub('', line).strip()
    # Extract index
    idx_m = INDEX_RE.match(line)
    if not idx_m:
        return None
    idx = idx_m.group(1)
    # Remove leading index from content
    content = line[idx_m.end():].strip()
    # Find type bracket
    t_m = TYPE_RE.search(content)
    typ = t_m.group(1) if t_m else None
    # Remove first type bracket from content if found
    if t_m:
        s = t_m.start()
        e = t_m.end()
        content = content[:s] + content[e:]
        content = content.strip()
    # Replace links with \href
    content = replace_links_with_href(content)
    # Append DOI href
    content = append_doi_href(content)
    # Escape LaTeX in content (after injecting \href and DOI)
    content_escaped = escape_text(content)
    # header with index and type
    if typ:
        header = f"［{idx}］［{typ}］ "
    else:
        header = f"［{idx}］ "
    # Compose final text
    final_text = header + content_escaped
    # Wrap into \bibitem
    return f"\\bibitem{{ref{idx}}} {final_text}"


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(IN_PATH)
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]
    items = []
    for ln in lines:
        if not ln.strip():
            continue
        bib = build_bibitem(ln)
        if bib:
            items.append(bib)
    n = len(items)
    header = (
        '% 参考文献列表（GB/T 7714—2015，保留全角编号［n］与类型标识）\n'
        '% 使用方法：\n'
        '% 1) 在主文档导言区加入：\\usepackage{hyperref}\n'
        '% 2) 在需要位置：\\input{output/ref_list_2023_2025_gbt7714.tex} 或复制下方 thebibliography 段落\n'
        '% 注：已转义 LaTeX 特殊字符；中文全角括号与标点保留；每条为一段；链接使用 \\href{URL}{URL}\n'
    )
    biblio_start = f"\\begin{{thebibliography}}{{{n}}}"
    biblio_end = "\\end{thebibliography}"
    out = header + "\n" + biblio_start + "\n" + "\n".join(items) + "\n" + biblio_end + "\n"
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(out)

if __name__ == '__main__':
    main()
