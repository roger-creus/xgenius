"""Simple web dashboard for inspecting xgenius DB.

Launch with: xgenius dashboard
Opens a local browser with tables for jobs, hypotheses, and stats.
Uses only Python stdlib (http.server + sqlite3) — no extra deps.
"""

import html
import json
import os
import sqlite3
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

from xgenius.config import load_config, get_xgenius_dir, get_run_id


DB_PATH = ""
CONFIG = None


def _query(sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _render_table(rows: list[dict], title: str = "", highlight_col: str = "") -> str:
    if not rows:
        return f"<h2>{title}</h2><p>No data.</p>"

    cols = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(c)}</th>" for c in cols)

    body = ""
    for row in rows:
        cells = ""
        for c in cols:
            val = str(row.get(c, ""))
            style = ""
            if c == highlight_col or c == "status":
                color_map = {
                    "completed": "#2ecc71", "failed": "#e74c3c", "running": "#3498db",
                    "pending": "#f39c12", "submitted": "#f39c12", "cancelled": "#95a5a6",
                    "disappeared": "#e67e22", "timeout": "#e74c3c", "oom": "#e74c3c",
                    "proposed": "#f39c12", "open": "#3498db", "promising": "#2ecc71", "closed": "#95a5a6",
                }
                bg = color_map.get(val.lower(), "")
                if bg:
                    style = f' style="background:{bg};color:white;padding:2px 8px;border-radius:4px"'
            cells += f"<td{style}>{html.escape(val[:100])}</td>"
        body += f"<tr>{cells}</tr>"

    return f"""
    <h2>{title}</h2>
    <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{body}</tbody>
    </table>
    """


def _render_page(content: str, nav: str = "") -> str:
    run_id = ""
    try:
        run_id = get_run_id(CONFIG) if CONFIG else ""
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>xgenius Dashboard</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #e94560; }}
        h2 {{ color: #0f3460; background: #16213e; padding: 10px; border-radius: 6px; color: #eee; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }}
        th {{ background: #16213e; color: #e94560; padding: 8px; text-align: left; position: sticky; top: 0; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #333; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        tr:hover {{ background: #16213e; }}
        nav {{ margin-bottom: 20px; }}
        nav a {{ color: #e94560; text-decoration: none; margin-right: 15px; font-weight: bold; }}
        nav a:hover {{ text-decoration: underline; }}
        .stats {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .stat {{ background: #16213e; padding: 15px 25px; border-radius: 8px; text-align: center; }}
        .stat .num {{ font-size: 28px; font-weight: bold; color: #e94560; }}
        .stat .label {{ font-size: 12px; color: #aaa; }}
        pre {{ background: #16213e; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>xgenius Dashboard {f'<small style="color:#666">run: {run_id}</small>' if run_id else ''}</h1>
    <nav>
        <a href="/">Overview</a>
        <a href="/jobs">All Jobs</a>
        <a href="/hypotheses">Hypotheses</a>
        <a href="/journal">Journal</a>
        <a href="/debug">Debug Log</a>
    </nav>
    {content}
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            content = self._overview()
        elif path == "/jobs":
            content = self._jobs(params)
        elif path == "/hypotheses":
            content = self._hypotheses()
        elif path == "/journal":
            content = self._journal()
        elif path == "/debug":
            content = self._debug()
        elif path == "/hypothesis":
            content = self._hypothesis_detail(params)
        else:
            content = "<h2>Not Found</h2>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_render_page(content).encode())

    def log_message(self, format, *args):
        pass  # Suppress request logs

    def _overview(self) -> str:
        total = _query("SELECT COUNT(*) as n FROM jobs")[0]["n"]
        by_status = _query("SELECT status, COUNT(*) as n FROM jobs GROUP BY status ORDER BY n DESC")
        hypotheses = _query("SELECT COUNT(*) as n FROM hypotheses")[0]["n"]

        stats = '<div class="stats">'
        stats += f'<div class="stat"><div class="num">{total}</div><div class="label">Total Jobs</div></div>'
        stats += f'<div class="stat"><div class="num">{hypotheses}</div><div class="label">Hypotheses</div></div>'
        for row in by_status:
            stats += f'<div class="stat"><div class="num">{row["n"]}</div><div class="label">{row["status"]}</div></div>'
        stats += '</div>'

        # Per-hypothesis summary
        hyp_rows = _query("""
            SELECT h.hypothesis_id, h.description, h.status as hyp_status,
                   COUNT(j.job_id) as total_jobs,
                   SUM(CASE WHEN j.status='completed' THEN 1 ELSE 0 END) as completed,
                   SUM(CASE WHEN j.status='failed' THEN 1 ELSE 0 END) as failed,
                   SUM(CASE WHEN j.status IN ('submitted','running','pending') THEN 1 ELSE 0 END) as active
            FROM hypotheses h
            LEFT JOIN jobs j ON h.hypothesis_id = j.hypothesis_id
            GROUP BY h.hypothesis_id
            ORDER BY h.created_at
        """)

        hyp_table = ""
        if hyp_rows:
            hyp_table = "<h2>Hypotheses</h2><table>"
            hyp_table += "<thead><tr><th>ID</th><th>Description</th><th>Status</th><th>Jobs</th><th>Done</th><th>Failed</th><th>Active</th></tr></thead><tbody>"
            for h in hyp_rows:
                hyp_table += f"""<tr>
                    <td><a href="/hypothesis?id={html.escape(h['hypothesis_id'])}" style="color:#e94560">{html.escape(h['hypothesis_id'])}</a></td>
                    <td>{html.escape(str(h['description'])[:80])}</td>
                    <td>{html.escape(str(h['hyp_status']))}</td>
                    <td>{h['total_jobs']}</td><td>{h['completed']}</td><td>{h['failed']}</td><td>{h['active']}</td>
                </tr>"""
            hyp_table += "</tbody></table>"

        recent = _query("SELECT job_id, experiment_id, cluster, status, walltime_seconds, exit_code FROM jobs ORDER BY submitted_at DESC LIMIT 10")
        recent_table = _render_table(recent, "Recent Jobs")

        return stats + hyp_table + recent_table

    def _jobs(self, params: dict) -> str:
        status_filter = params.get("status", [None])[0]
        hyp_filter = params.get("hypothesis_id", [None])[0]

        # Filter links
        filters = '<p>Filter: <a href="/jobs">All</a>'
        for s in ["submitted", "running", "completed", "failed", "cancelled", "disappeared"]:
            filters += f' | <a href="/jobs?status={s}">{s}</a>'
        filters += '</p>'

        sql = "SELECT * FROM jobs WHERE 1=1"
        params_list = []
        if status_filter:
            sql += " AND status=?"
            params_list.append(status_filter)
        if hyp_filter:
            sql += " AND hypothesis_id=?"
            params_list.append(hyp_filter)
        sql += " ORDER BY submitted_at DESC LIMIT 500"

        rows = _query(sql, tuple(params_list))
        title = f"Jobs ({len(rows)})"
        if status_filter:
            title += f" — {status_filter}"
        return filters + _render_table(rows, title)

    def _hypotheses(self) -> str:
        rows = _query("SELECT * FROM hypotheses ORDER BY created_at")
        return _render_table(rows, "All Hypotheses")

    def _hypothesis_detail(self, params: dict) -> str:
        hid = params.get("id", [None])[0]
        if not hid:
            return "<h2>Missing hypothesis ID</h2>"

        hyp = _query("SELECT * FROM hypotheses WHERE hypothesis_id=?", (hid,))
        jobs = _query("SELECT * FROM jobs WHERE hypothesis_id=? ORDER BY submitted_at", (hid,))

        content = ""
        if hyp:
            h = hyp[0]
            content += f"<h2>Hypothesis: {html.escape(hid)}</h2>"
            content += f"<p><b>Description:</b> {html.escape(str(h.get('description', '')))}</p>"
            content += f"<p><b>Status:</b> {html.escape(str(h.get('status', '')))}</p>"
            content += f"<p><b>Motivation:</b> {html.escape(str(h.get('motivation', '')))}</p>"
            content += f"<p><b>Conclusion:</b> {html.escape(str(h.get('conclusion', '')))}</p>"

        content += _render_table(jobs, f"Jobs for {hid}")
        return content

    def _journal(self) -> str:
        xgenius_dir = get_xgenius_dir(CONFIG)
        journal_path = os.path.join(xgenius_dir, "journal.md")
        if os.path.exists(journal_path):
            with open(journal_path) as f:
                content = f.read()
            return f"<h2>Research Journal</h2><pre>{html.escape(content)}</pre>"
        return "<h2>Research Journal</h2><p>Empty.</p>"

    def _debug(self) -> str:
        xgenius_dir = get_xgenius_dir(CONFIG)
        debug_path = os.path.join(xgenius_dir, "DEBUG.md")
        if os.path.exists(debug_path):
            with open(debug_path) as f:
                content = f.read()
            return f"<h2>Debug Log</h2><pre>{html.escape(content)}</pre>"
        return "<h2>Debug Log</h2><p>No errors logged.</p>"


def run_dashboard(config_path: str = "xgenius.toml", port: int = 8765) -> None:
    """Start the xgenius dashboard web server."""
    global DB_PATH, CONFIG
    CONFIG = load_config(config_path)
    xgenius_dir = get_xgenius_dir(CONFIG)
    DB_PATH = os.path.join(xgenius_dir, "xgenius.db")

    server = HTTPServer(("localhost", port), DashboardHandler)
    print(f"xgenius dashboard: http://localhost:{port}")
    print(f"DB: {DB_PATH}")
    print("Press Ctrl+C to stop.")

    import webbrowser
    webbrowser.open(f"http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
