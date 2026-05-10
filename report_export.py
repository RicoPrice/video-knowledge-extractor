"""报告批量导出：Markdown → PDF（weasyprint） + ZIP 打包

该模块被 app.py 调用。数据库里存的 report_markdown 可能包含相对路径的图片引用
（例如 ./screenshots/xxx.jpg），需要在渲染 HTML 时指定 base_url 让 weasyprint
能够加载这些本地资源。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import markdown as md_lib

# weasyprint 首次 import 较慢，放在模块顶层一次性加载
from weasyprint import HTML  # type: ignore

PDF_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; color: #222; font-size: 11pt; line-height: 1.6; }
h1, h2, h3, h4 { color: #1e293b; margin: 1em 0 .4em; }
h1 { font-size: 20pt; border-bottom: 2px solid #4f46e5; padding-bottom: .3em; }
h2 { font-size: 15pt; border-bottom: 1px solid #e2e8f0; padding-bottom: .25em; }
h3 { font-size: 13pt; }
p { margin: .4em 0; }
ul, ol { padding-left: 1.4em; }
li { margin: .2em 0; }
code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px; font-size: 10pt; }
pre { background: #0f172a; color: #f1f5f9; padding: 10px; border-radius: 4px; overflow-wrap: break-word; white-space: pre-wrap; }
blockquote { border-left: 3px solid #4f46e5; background: #eef2ff; padding: .2em .8em; color: #334155; margin: .5em 0; }
img { max-width: 100%; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: .6em 0; }
th, td { border: 1px solid #cbd5e1; padding: 5px 8px; font-size: 10pt; }
th { background: #f1f5f9; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1em 0; }
a { color: #4f46e5; text-decoration: none; }
"""


def render_pdf(markdown_text: str, base_dir: Path, title: str = "报告") -> bytes:
    """把 markdown 文本渲染成 PDF bytes。base_dir 用于解析 md 里的相对图片路径。"""
    html_body = md_lib.markdown(
        markdown_text or "",
        extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
    )
    html_doc = (
        f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{_escape(title)}</title><style>{PDF_CSS}</style></head>"
        f"<body><h1>{_escape(title)}</h1>{html_body}</body></html>"
    )
    return HTML(string=html_doc, base_url=str(base_dir)).write_pdf()


def _escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_filename(name: str, max_len: int = 120) -> str:
    """清理成可放入 zip 的文件名（保留中文，替换路径分隔符）。"""
    cleaned = _UNSAFE_NAME_RE.sub("_", (name or "report")).strip().strip(".")
    if not cleaned:
        cleaned = "report"
    return cleaned[:max_len]


FMT_SPEC = {
    # fmt: (数据库字段, 文件后缀, 文件名后缀, 是否需要合成 PDF)
    "pdf": ("report_markdown", ".pdf", "", True),
    "md": ("report_markdown", ".md", "", False),
    "srt": ("report_srt", ".srt", "_总结", False),
    "raw_srt": ("raw_srt", ".srt", "_原始字幕", False),
}


def build_zip(
    tasks: Iterable[dict], fmt: str, output_root: Path
) -> tuple[bytes, int, list[str]]:
    """打包多个任务的导出产物为 zip。

    返回 (zip_bytes, 成功文件数, 跳过的任务名列表)。
    同名任务会自动追加 _2 / _3 后缀避免覆盖。
    """
    if fmt not in FMT_SPEC:
        raise ValueError(f"unsupported fmt: {fmt}")
    field, ext, suffix, is_pdf = FMT_SPEC[fmt]

    buf = io.BytesIO()
    success = 0
    skipped: list[str] = []
    used_names: set[str] = set()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for task in tasks:
            name = task.get("video_name") or task.get("id", "report")
            raw = task.get(field) or ""
            if not raw:
                skipped.append(name)
                continue

            base_name = safe_filename(f"{name}{suffix}")
            final_name = _dedupe(base_name, ext, used_names)

            if is_pdf:
                try:
                    pdf_bytes = render_pdf(
                        raw,
                        base_dir=output_root / (task.get("video_name") or ""),
                        title=name,
                    )
                except Exception:
                    skipped.append(name)
                    continue
                zf.writestr(final_name, pdf_bytes)
            else:
                zf.writestr(final_name, raw.encode("utf-8"))
            success += 1

    return buf.getvalue(), success, skipped


def _dedupe(base: str, ext: str, used: set[str]) -> str:
    candidate = f"{base}{ext}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    i = 2
    while True:
        candidate = f"{base}_{i}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1
