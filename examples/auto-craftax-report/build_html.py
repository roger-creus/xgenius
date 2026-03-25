#!/usr/bin/env python3
"""Convert report.md to a self-contained HTML file with embedded base64 images."""

import base64
import os
import re
import markdown

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(REPORT_DIR, 'report.md')
HTML_PATH = os.path.join(REPORT_DIR, 'report.html')
PLOTS_DIR = os.path.join(REPORT_DIR, 'plots')

# Read markdown
with open(MD_PATH, 'r') as f:
    md_content = f.read()

# Replace image references with base64 data URIs
def embed_image(match):
    alt = match.group(1)
    path = match.group(2)
    full_path = os.path.join(REPORT_DIR, path)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as img_file:
            b64 = base64.b64encode(img_file.read()).decode('utf-8')
        ext = os.path.splitext(path)[1].lower()
        mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'svg': 'image/svg+xml'}.get(ext.lstrip('.'), 'image/png')
        return f'![{alt}](data:{mime};base64,{b64})'
    return match.group(0)

md_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed_image, md_content)

# Convert to HTML
html_body = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc'],
    extension_configs={
        'codehilite': {'css_class': 'highlight'},
    }
)

# CSS styling
CSS = """
:root {
    --bg: #ffffff;
    --text: #1a1a2e;
    --heading: #16213e;
    --accent: #0f3460;
    --border: #e0e0e0;
    --code-bg: #f5f5f5;
    --table-header: #1a1a2e;
    --table-stripe: #f8f9fa;
    --link: #0f3460;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    font-size: 16px;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg);
    max-width: 960px;
    margin: 0 auto;
    padding: 40px 30px 80px;
}

h1 {
    font-size: 2em;
    color: var(--heading);
    border-bottom: 3px solid var(--accent);
    padding-bottom: 12px;
    margin: 40px 0 20px;
    line-height: 1.3;
}

h1:first-child { margin-top: 0; }

h2 {
    font-size: 1.5em;
    color: var(--heading);
    border-bottom: 2px solid var(--border);
    padding-bottom: 8px;
    margin: 36px 0 16px;
}

h3 {
    font-size: 1.2em;
    color: var(--accent);
    margin: 28px 0 12px;
}

h4 {
    font-size: 1.05em;
    color: var(--accent);
    margin: 20px 0 8px;
}

p { margin: 0 0 16px; }

strong { color: var(--heading); }

a { color: var(--link); text-decoration: none; border-bottom: 1px solid transparent; }
a:hover { border-bottom-color: var(--link); }

hr {
    border: none;
    border-top: 2px solid var(--border);
    margin: 32px 0;
}

img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin: 16px 0;
    display: block;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 24px;
    font-size: 14px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

thead th {
    background: var(--table-header);
    color: #fff;
    font-weight: 600;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

tbody td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
}

tbody tr:nth-child(even) { background: var(--table-stripe); }
tbody tr:hover { background: #eef2ff; }

code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.88em;
    background: var(--code-bg);
    padding: 2px 6px;
    border-radius: 4px;
    color: #c7254e;
}

pre {
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 20px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 16px 0 24px;
    font-size: 13px;
    line-height: 1.6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12);
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
}

ul, ol {
    margin: 0 0 16px 24px;
}

li { margin-bottom: 6px; }
li > ul, li > ol { margin-top: 6px; margin-bottom: 0; }

blockquote {
    border-left: 4px solid var(--accent);
    background: var(--table-stripe);
    padding: 12px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
    color: #555;
}

em { color: #555; }

/* Print styles */
@media print {
    body { max-width: 100%; padding: 20px; font-size: 12px; }
    h1 { font-size: 1.6em; }
    h2 { font-size: 1.3em; }
    img { box-shadow: none; }
    pre { font-size: 11px; }
}

/* Responsive */
@media (max-width: 768px) {
    body { padding: 20px 16px; font-size: 15px; }
    h1 { font-size: 1.6em; }
    table { font-size: 12px; }
    thead th, tbody td { padding: 6px 8px; }
}
"""

# Full HTML document
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Craftax-Symbolic-v1 Research Report</title>
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(HTML_PATH, 'w') as f:
    f.write(html)

print(f"HTML report generated: {HTML_PATH}")
print(f"File size: {os.path.getsize(HTML_PATH) / 1024 / 1024:.1f} MB")
