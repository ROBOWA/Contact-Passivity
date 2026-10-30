"""Minimal Markdown -> HTML -> PDF (headless Google Chrome). Supports headings, paragraphs, lists,
pipe tables, fenced/indented code, inline code, bold/italic, links and images (no math rendering;
documents use plain Unicode math)."""
import html
import re
import subprocess
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
body{font-family:-apple-system,"PingFang SC","Helvetica Neue",Arial,sans-serif;font-size:10.5pt;line-height:1.45;
     color:#0b0b0b;max-width:1000px;margin:0 auto;padding:12px 18px;background:#fff}
h1{font-size:19pt;border-bottom:1px solid #c3c2b7;padding-bottom:4px} h2{font-size:14.5pt;margin-top:20px;
border-bottom:1px solid #e1e0d9} h3{font-size:12pt} h4{font-size:11pt}
code,pre{font-family:Menlo,Consolas,monospace;font-size:9pt} pre{background:#f6f6f4;padding:8px;overflow-x:auto;
white-space:pre-wrap;border:1px solid #e1e0d9;border-radius:4px} code{background:#f3f3f0;padding:0 2px;border-radius:2px}
table{border-collapse:collapse;margin:8px 0;font-size:8.6pt} th,td{border:1px solid #d6d5ce;padding:3px 5px;
vertical-align:top} th{background:#f1f0ec} img{max-width:100%;margin:6px 0;border:1px solid #eee}
blockquote{border-left:3px solid #c3c2b7;margin:6px 0;padding:2px 10px;color:#52514e}
@page{size:A4;margin:14mm 12mm}
"""


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?!\w)", r"<i>\1</i>", s)
    return s


def md_to_html(text, title="document"):
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
            i = j + 1
            continue
        if ln.startswith("    ") and ln.strip():
            buf = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                buf.append(lines[i][4:])
                i += 1
            while buf and not buf[-1].strip():
                buf.pop()
            out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        if ln.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(re.match(r"^:?-+:?$", c) for c in r if c)]
            t = "<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
            for r in body:
                t += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</table>")
            continue
        if re.match(r"^\s*([*-]|\d+\.)\s+", ln):
            items = []          # (indent, ordered, text); nesting by indentation
            while i < len(lines) and (re.match(r"^\s*([*-]|\d+\.)\s+", lines[i]) or (lines[i].startswith("  ") and lines[i].strip() and items)):
                m2 = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)", lines[i])
                if m2:
                    items.append([len(m2.group(1)), m2.group(2)[0].isdigit(), m2.group(3)])
                else:
                    items[-1][2] += " " + lines[i].strip()
                i += 1
            html_, stack = "", []
            for ind, ordered, text in items:
                tag = "ol" if ordered else "ul"
                if not stack or ind > stack[-1][0]:
                    html_ += f"<{tag}>"
                    stack.append((ind, tag))
                else:
                    while len(stack) > 1 and ind < stack[-1][0]:
                        html_ += f"</li></{stack[-1][1]}>"
                        stack.pop()
                    html_ += "</li>"
                html_ += f"<li>{inline(text)}"
            while stack:
                html_ += f"</li></{stack[-1][1]}>"
                stack.pop()
            out.append(html_)
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        if not ln.strip():
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\||```|\s*([*-]|\d+\.)\s+|>)", lines[i]) \
                and not lines[i].startswith("    "):
            buf.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>" + "\n".join(out) + "</body></html>")


def md_to_pdf(md_path, pdf_path=None):
    md_path = Path(md_path)
    pdf_path = Path(pdf_path or md_path.with_suffix(".pdf"))
    html_path = md_path.with_suffix(".html")
    html_path.write_text(md_to_html(md_path.read_text(), md_path.stem))
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
           f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return pdf_path if pdf_path.exists() else (r.stdout + r.stderr)
