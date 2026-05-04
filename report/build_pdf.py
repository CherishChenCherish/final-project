"""
Convert REPORT.md to REPORT.pdf using pandoc + weasyprint.
Standalone — no external TeX install required.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, 'REPORT.md')
HTML = os.path.join(HERE, 'REPORT.html')
PDF = os.path.join(HERE, 'REPORT.pdf')

CSS = """
@page {
    size: letter;
    margin: 0.85in 0.85in 1in 0.85in;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}
body {
    font-family: -apple-system, "Helvetica Neue", "Segoe UI", "Liberation Sans", sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #111;
    max-width: 100%;
}
h1 {
    font-size: 17pt;
    margin-top: 0;
    border-bottom: 2px solid #222;
    padding-bottom: 4pt;
}
h2 {
    font-size: 13pt;
    margin-top: 18pt;
    color: #222;
    border-bottom: 1px solid #ccc;
    padding-bottom: 2pt;
}
h3 {
    font-size: 11.5pt;
    color: #333;
    margin-top: 12pt;
}
p { margin: 6pt 0; }
ul, ol { margin: 6pt 0; padding-left: 22pt; }
li { margin: 1pt 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #aaa;
    padding: 3pt 6pt;
    text-align: left;
}
th { background: #eee; font-weight: 600; }
code {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 1pt 3pt;
    border-radius: 2pt;
    font-family: "SF Mono", Monaco, Consolas, monospace;
    font-size: 9pt;
}
pre {
    background: #f8f8f8;
    border: 1px solid #ddd;
    padding: 6pt;
    border-radius: 3pt;
    font-size: 9pt;
    overflow-x: auto;
}
pre code { background: transparent; border: 0; padding: 0; }
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 6pt auto;
    page-break-inside: avoid;
}
blockquote {
    border-left: 3px solid #888;
    padding-left: 10pt;
    color: #444;
    margin: 6pt 0;
    font-style: italic;
}
hr {
    border: 0;
    border-top: 1px solid #ccc;
    margin: 14pt 0;
}
strong { color: #000; }
"""


def main():
    # 1. Markdown → HTML via pandoc
    print("Converting Markdown to HTML...")
    subprocess.run([
        'pandoc', MD,
        '-t', 'html5', '-s',
        '--metadata', 'title=BTC Market Regime Classification',
        '-o', HTML,
    ], check=True)

    # 2. Inject our CSS into the HTML
    with open(HTML, 'r') as f:
        html_text = f.read()
    if '<head>' in html_text:
        html_text = html_text.replace(
            '<head>',
            f'<head>\n<style>{CSS}</style>\n',
            1,
        )
    with open(HTML, 'w') as f:
        f.write(html_text)

    # 3. HTML → PDF via WeasyPrint
    print("Rendering HTML to PDF (this may take ~10s)...")
    os.environ.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/lib')
    from weasyprint import HTML as WPHTML
    WPHTML(filename=HTML, base_url=HERE).write_pdf(PDF)

    size_kb = os.path.getsize(PDF) / 1024
    print(f"  Wrote {PDF} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
