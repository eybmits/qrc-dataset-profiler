"""Local web interface for QRC-usefulness triage.

The web app is intentionally dependency-light: it uses Python's standard HTTP
server and the existing triage module. The browser sends CSV text to a local
JSON endpoint; the server computes atlas descriptors and returns the same
bounded triage report as the command-line tool.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrc_dataset_profiler.triage import (
    DEFAULT_DISCOVERY_TABLE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_WIN_THRESHOLD,
    feature_row_from_series,
    fit_triage_model,
    load_discovery_table,
    report_to_text,
    series_from_dataframe,
    triage_feature_row,
)


MAX_REQUEST_BYTES = 8 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local web UI for QRC-usefulness triage.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on.")
    parser.add_argument("--discovery-table", default=str(DEFAULT_DISCOVERY_TABLE), help="Frozen discovery table used by the triage model.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic meta-model seed.")
    parser.add_argument("--win-threshold", type=float, default=DEFAULT_WIN_THRESHOLD, help="QRC-useful advantage threshold.")
    args = parser.parse_args(argv)

    discovery = load_discovery_table(Path(args.discovery_table))
    model = fit_triage_model(discovery, seed=args.seed, win_threshold=args.win_threshold)
    handler = _handler_factory(discovery_table=discovery, model=model)
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"QRC triage web UI running at http://{args.host}:{args.port}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _handler_factory(discovery_table: pd.DataFrame, model: Any) -> type[BaseHTTPRequestHandler]:
    class TriageHandler(BaseHTTPRequestHandler):
        server_version = "QRCTriageWeb/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_text(_html_page(), content_type="text/html; charset=utf-8")
                return
            if self.path == "/healthz":
                self._send_json({"ok": True, "n_discovery": model.n_discovery})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_HEAD(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                data = _html_page().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return
            if self.path == "/healthz":
                data = json.dumps({"ok": True, "n_discovery": model.n_discovery}).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/triage":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                self._send_json({"error": "empty request body"}, status=HTTPStatus.BAD_REQUEST)
                return
            if length > MAX_REQUEST_BYTES:
                self._send_json({"error": "request too large; use a smaller CSV window"}, status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                report = _report_from_payload(payload, discovery_table=discovery_table, model=model)
            except Exception as exc:  # pragma: no cover - exercised through browser/manual use.
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(report)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _send_text(self, text: str, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(_json_safe(payload), indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return TriageHandler


def _report_from_payload(payload: dict[str, Any], *, discovery_table: pd.DataFrame, model: Any) -> dict[str, Any]:
    csv_text = str(payload.get("csv_text", "")).strip()
    if not csv_text:
        raise ValueError("paste or upload a CSV time series first")
    sep = _parse_sep(payload.get("sep", ","))
    header = bool(payload.get("header", True))
    column = _parse_column(payload.get("column"))
    name = str(payload.get("name") or "web_series")
    horizon = int(payload.get("horizon") or 1)
    max_length = int(payload.get("max_length") or DEFAULT_MAX_LENGTH)
    window = str(payload.get("window") or "tail")
    if window not in {"tail", "head", "full"}:
        raise ValueError("window must be 'tail', 'head', or 'full'")

    df = pd.read_csv(StringIO(csv_text), sep=sep, header=0 if header else None)
    series = series_from_dataframe(df, column=column, source_label="uploaded CSV", header=header)
    features = feature_row_from_series(series, name=name, horizon=horizon, max_length=max_length, window=window)
    report = triage_feature_row(features, model, discovery_table=discovery_table)
    report["dataset"] = {
        "raw_length": int(np.asarray(series).reshape(-1).size),
        "used_length": int(features.loc[0, "triage_used_length"]) if "triage_used_length" in features else int(features.loc[0, "length"]),
        "window": window,
        "horizon": horizon,
    }
    report["text_report"] = report_to_text(report)
    return report


def _parse_sep(value: Any) -> str:
    raw = str(value or ",")
    if raw in {"tab", "\\t", "\t"}:
        return "\t"
    return raw


def _parse_column(value: Any) -> str | int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _html_page() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QRC Regime Triage</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #657185;
      --line: #d8dee8;
      --paper: #f7f8fb;
      --panel: #ffffff;
      --blue: #2f62b7;
      --teal: #1f8a83;
      --amber: #b87513;
      --red: #b94a48;
      --green: #2d7d46;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 24px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.02;
      letter-spacing: 0;
    }
    .sub {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
      max-width: 720px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(340px, 0.92fr) minmax(420px, 1.08fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 16px 36px rgba(23, 32, 42, 0.06);
    }
    .panel h2 {
      margin: 0;
      padding: 18px 18px 10px;
      font-size: 18px;
      line-height: 1.2;
    }
    .controls {
      padding: 0 18px 18px;
      display: grid;
      gap: 13px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 650;
      color: #2f3a4d;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      padding: 10px 11px;
      outline: none;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(47, 98, 183, 0.13);
    }
    textarea {
      min-height: 230px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      line-height: 1.45;
    }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    .checkline {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: var(--ink);
    }
    .checkline input { width: 17px; height: 17px; }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding-top: 2px;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
      background: var(--blue);
      color: #fff;
    }
    button.secondary {
      background: #e9edf5;
      color: #1e2b3f;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .result {
      min-height: 620px;
      padding: 18px;
    }
    .empty {
      height: 580px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
    }
    .statusbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 16px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 800;
      color: #fff;
      background: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }
    .badge.high { background: var(--green); }
    .badge.worth { background: var(--teal); }
    .badge.low { background: var(--amber); }
    .badge.ood { background: var(--red); }
    .metricgrid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      min-height: 82px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric .value {
      margin-top: 8px;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .section {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }
    .section h3 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .feature {
      display: grid;
      grid-template-columns: minmax(130px, 170px) 1fr minmax(62px, max-content);
      gap: 10px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      background: #e6ebf3;
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--teal), var(--blue));
    }
    pre {
      white-space: pre-wrap;
      background: #111827;
      color: #edf2f7;
      border-radius: 8px;
      padding: 13px;
      font-size: 12px;
      line-height: 1.45;
      overflow: auto;
    }
    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .error {
      border: 1px solid rgba(185, 74, 72, 0.35);
      background: rgba(185, 74, 72, 0.08);
      color: #7f2523;
      border-radius: 8px;
      padding: 12px;
      font-weight: 650;
    }
    @media (max-width: 900px) {
      main { width: min(100vw - 20px, 760px); padding-top: 18px; }
      header { display: block; }
      .layout { grid-template-columns: 1fr; }
      .metricgrid { grid-template-columns: 1fr; }
      .grid3 { grid-template-columns: 1fr; }
      .feature { grid-template-columns: 1fr; }
      .result { min-height: 0; }
      .empty { height: 280px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>QRC Regime Triage</h1>
      <p class="sub">Screen a univariate time series against the frozen v5 QRC-vs-ESN atlas.</p>
    </div>
  </header>

  <div class="layout">
    <section class="panel">
      <h2>Dataset</h2>
      <form id="triageForm" class="controls">
        <label>CSV file
          <input id="fileInput" type="file" accept=".csv,.txt,.tsv,text/csv,text/plain">
        </label>
        <label>CSV text
          <textarea id="csvText" spellcheck="false" placeholder="value&#10;1.02&#10;1.08&#10;1.05"></textarea>
        </label>
        <div class="grid2">
          <label>Dataset name
            <input id="nameInput" value="my_dataset">
          </label>
          <label>Target column
            <input id="columnInput" value="value">
          </label>
        </div>
        <div class="grid3">
          <label>Separator
            <select id="sepInput">
              <option value=",">Comma</option>
              <option value=";">Semicolon</option>
              <option value="tab">Tab</option>
            </select>
          </label>
          <label>Window
            <select id="windowInput">
              <option value="tail">Tail</option>
              <option value="head">Head</option>
              <option value="full">Full</option>
            </select>
          </label>
          <label>Max length
            <input id="maxLengthInput" type="number" min="0" step="1" value="800">
          </label>
        </div>
        <label class="checkline">
          <input id="headerInput" type="checkbox" checked>
          Header row
        </label>
        <div class="actions">
          <button id="runButton" type="submit">Run Triage</button>
          <button id="demoButton" class="secondary" type="button">Load Demo</button>
          <button id="clearButton" class="secondary" type="button">Clear</button>
        </div>
        <p class="small">The tool computes descriptors and a triage score. It does not run QRC or ESN on the uploaded series.</p>
      </form>
    </section>

    <section class="panel result" id="resultPanel">
      <div class="empty">Run a dataset to see the recommendation, atlas support, and descriptor position.</div>
    </section>
  </div>
</main>

<script>
const form = document.getElementById("triageForm");
const resultPanel = document.getElementById("resultPanel");
const runButton = document.getElementById("runButton");
const csvText = document.getElementById("csvText");
const fileInput = document.getElementById("fileInput");

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  csvText.value = await file.text();
  const base = file.name.replace(/\.[^.]+$/, "");
  if (base) document.getElementById("nameInput").value = base;
});

document.getElementById("demoButton").addEventListener("click", () => {
  const rows = ["value"];
  let level = 0.0;
  for (let i = 0; i < 900; i += 1) {
    level += 0.01 + 0.03 * Math.sin(i / 40) + 0.02 * Math.sin(i / 9);
    const value = level + 0.12 * Math.sin(i / 25) + 0.04 * Math.sin(i * 1.7);
    rows.push(value.toFixed(6));
  }
  csvText.value = rows.join("\n");
  document.getElementById("nameInput").value = "slow_stateful_demo";
  document.getElementById("columnInput").value = "value";
  document.getElementById("headerInput").checked = true;
});

document.getElementById("clearButton").addEventListener("click", () => {
  csvText.value = "";
  fileInput.value = "";
  resultPanel.innerHTML = '<div class="empty">Run a dataset to see the recommendation, atlas support, and descriptor position.</div>';
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  runButton.disabled = true;
  resultPanel.innerHTML = '<div class="empty">Computing descriptors and atlas support...</div>';
  const payload = {
    csv_text: csvText.value,
    column: document.getElementById("columnInput").value,
    sep: document.getElementById("sepInput").value,
    header: document.getElementById("headerInput").checked,
    name: document.getElementById("nameInput").value,
    window: document.getElementById("windowInput").value,
    max_length: Number(document.getElementById("maxLengthInput").value || 800),
    horizon: 1
  };
  try {
    const response = await fetch("/api/triage", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "triage failed");
    renderReport(data);
  } catch (err) {
    resultPanel.innerHTML = `<div class="error">${escapeHtml(err.message || String(err))}</div>`;
  } finally {
    runButton.disabled = false;
  }
});

function renderReport(report) {
  const pred = report.prediction || {};
  const support = report.support || {};
  const features = report.key_features || {};
  const rec = pred.recommendation || "unknown";
  const badgeClass = rec.includes("high") ? "high" : rec.includes("worth") ? "worth" : rec.includes("outside") ? "ood" : "low";
  const featureRows = Object.entries(features).map(([name, payload]) => {
    const pct = Math.max(0, Math.min(1, Number(payload.atlas_percentile || 0)));
    return `<div class="feature">
      <div>${formatFeatureName(name)}</div>
      <div class="bar"><span style="width:${(pct * 100).toFixed(1)}%"></span></div>
      <div>${(pct * 100).toFixed(0)}%</div>
    </div>`;
  }).join("");
  resultPanel.innerHTML = `
    <div class="statusbar">
      <div>
        <div class="badge ${badgeClass}">${escapeHtml(rec.replaceAll("_", " "))}</div>
        <div class="small" style="margin-top:8px">${escapeHtml(report.name || "dataset")}</div>
      </div>
      <div class="small">${escapeHtml((report.dataset && report.dataset.used_length) ? String(report.dataset.used_length) : "?")} samples used</div>
    </div>
    <div class="metricgrid">
      <div class="metric"><div class="label">Predicted advantage</div><div class="value">${formatSigned(pred.predicted_best_qrc_advantage_vs_esn)}</div></div>
      <div class="metric"><div class="label">Useful probability</div><div class="value">${formatPct(pred.qrc_useful_probability)}</div></div>
      <div class="metric"><div class="label">Atlas support</div><div class="value">${formatPct(support.support_score)}</div></div>
    </div>
    <div class="section">
      <h3>Nearest atlas families</h3>
      <p class="small">${escapeHtml(support.nearest_family_mixture || "not available")} ${support.ood_flag ? "(OOD)" : ""}</p>
    </div>
    <div class="section">
      <h3>Descriptor position</h3>
      ${featureRows || '<p class="small">No key descriptor percentiles available.</p>'}
    </div>
    <div class="section">
      <h3>Boundary</h3>
      <p class="small">${escapeHtml(report.claim_boundary || "")}</p>
    </div>
    <div class="section">
      <h3>Text report</h3>
      <pre>${escapeHtml(report.text_report || "")}</pre>
    </div>`;
}

function formatPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${(num * 100).toFixed(1)}%`;
}

function formatSigned(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${num >= 0 ? "+" : ""}${num.toFixed(3)}`;
}

function formatFeatureName(name) {
  return name.replace(/^ext_/, "").replaceAll("_", " ");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
