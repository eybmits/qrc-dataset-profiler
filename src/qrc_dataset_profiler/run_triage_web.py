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
    .metric .meaning {
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
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
    .callout {
      margin: 0 0 16px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      border-radius: 8px;
      padding: 13px 14px;
      background: #fbfcff;
    }
    .callout.high { border-left-color: var(--green); }
    .callout.worth { border-left-color: var(--teal); }
    .callout.low { border-left-color: var(--amber); }
    .callout.ood { border-left-color: var(--red); }
    .callout h3 {
      margin: 0 0 6px;
      font-size: 18px;
    }
    .callout p {
      margin: 6px 0 0;
      color: #344054;
      font-size: 14px;
      line-height: 1.45;
    }
    .markers {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .markers li {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      font-size: 13px;
      line-height: 1.35;
      background: #fff;
    }
    .markers strong { color: var(--ink); }
    .feature {
      display: grid;
      grid-template-columns: minmax(130px, 170px) 1fr minmax(62px, max-content);
      gap: 10px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }
    .feature .hint {
      color: var(--muted);
      font-size: 12px;
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
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
    }
    summary {
      cursor: pointer;
      font-weight: 750;
      font-size: 14px;
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
      <p class="sub">Upload a time series and get a plain-language priority signal: should QRC be tested, or is ESN the better default?</p>
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
          <button id="runButton" type="submit">Check QRC Priority</button>
          <button id="demoButton" class="secondary" type="button">Load Demo</button>
          <button id="clearButton" class="secondary" type="button">Clear</button>
        </div>
        <p class="small">The tool looks for dataset markers that were associated with QRC gains in the atlas. It does not prove that QRC will win.</p>
      </form>
    </section>

    <section class="panel result" id="resultPanel">
      <div class="empty">Run a dataset to see whether QRC should be prioritized and which markers support that decision.</div>
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
  resultPanel.innerHTML = '<div class="empty">Checking QRC markers and similarity to known examples...</div>';
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
  const recInfo = recommendationInfo(rec);
  const markers = buildMarkers(features, pred, support);
  const featureRows = Object.entries(features).map(([name, payload]) => {
    const pct = Math.max(0, Math.min(1, Number(payload.atlas_percentile || 0)));
    return `<div class="feature">
      <div><strong>${formatFeatureName(name)}</strong><div class="hint">${featureMeaning(name, pct)}</div></div>
      <div class="bar"><span style="width:${(pct * 100).toFixed(1)}%"></span></div>
      <div>${percentileLabel(pct)}</div>
    </div>`;
  }).join("");
  resultPanel.innerHTML = `
    <div class="statusbar">
      <div>
        <div class="badge ${badgeClass}">${escapeHtml(recInfo.badge)}</div>
        <div class="small" style="margin-top:8px">${escapeHtml(report.name || "dataset")} · ${escapeHtml(recInfo.action)}</div>
      </div>
      <div class="small">${escapeHtml((report.dataset && report.dataset.used_length) ? String(report.dataset.used_length) : "?")} samples used</div>
    </div>
    <div class="callout ${badgeClass}">
      <h3>${escapeHtml(recInfo.title)}</h3>
      <p>${escapeHtml(recInfo.body)}</p>
    </div>
    <div class="metricgrid">
      <div class="metric"><div class="label">Expected QRC gain</div><div class="value">${formatSigned(pred.predicted_best_qrc_advantage_vs_esn)}</div><div class="meaning">Positive means the model expects QRC to reduce error versus ESN.</div></div>
      <div class="metric"><div class="label">QRC priority score</div><div class="value">${formatPct(pred.qrc_useful_probability)}</div><div class="meaning">Higher means stronger evidence that this is a QRC-worthy regime.</div></div>
      <div class="metric"><div class="label">Similarity to known cases</div><div class="value">${formatPct(support.support_score)}</div><div class="meaning">Low similarity means the tool should not be trusted without direct benchmarking.</div></div>
    </div>
    <div class="section">
      <h3>Markers for QRC advantage</h3>
      <ul class="markers">${markers.map(m => `<li><strong>${escapeHtml(m.title)}</strong><br>${escapeHtml(m.body)}</li>`).join("")}</ul>
    </div>
    <div class="section">
      <h3>Looks most like</h3>
      <p class="small">${escapeHtml(formatFamilies(support.nearest_family_mixture) || "not available")} ${support.ood_flag ? "This is outside the familiar atlas region, so direct benchmarking is safer." : ""}</p>
    </div>
    <div class="section">
      <h3>Dataset marker positions</h3>
      ${featureRows || '<p class="small">No key descriptor percentiles available.</p>'}
    </div>
    <div class="section">
      <h3>Important boundary</h3>
      <p class="small">${escapeHtml(report.claim_boundary || "")}</p>
    </div>
    <div class="section">
      <details>
        <summary>Show technical report</summary>
        <pre>${escapeHtml(report.text_report || "")}</pre>
      </details>
    </div>`;
}

function recommendationInfo(rec) {
  if (rec === "high_priority_qrc_test") {
    return {
      badge: "High QRC priority",
      title: "Test QRC for this dataset.",
      body: "The dataset matches patterns where QRC was often useful against the frozen ESN baseline.",
      action: "Run the QRC-vs-ESN benchmark next"
    };
  }
  if (rec === "worth_testing_qrc_if_available") {
    return {
      badge: "QRC worth testing",
      title: "QRC is plausible, but not guaranteed.",
      body: "The dataset has some markers seen in QRC-favorable regimes. ESN remains a strong default, but QRC is worth a direct test.",
      action: "Benchmark QRC if available"
    };
  }
  if (rec === "outside_atlas_support__run_direct_benchmark") {
    return {
      badge: "Benchmark directly",
      title: "This dataset is too different from the atlas.",
      body: "The tool found low similarity to known atlas examples. Do not rely on the prediction alone; run ESN and QRC directly.",
      action: "Do not trust the score alone"
    };
  }
  return {
    badge: "ESN first",
    title: "QRC is low priority for this dataset.",
    body: "The atlas suggests that a standard ESN is the better first model unless you have another scientific reason to test QRC.",
    action: "Use ESN as the default"
  };
}

function buildMarkers(features, pred, support) {
  const markers = [];
  const trend = percentileOf(features.ext_trend_strength);
  const centroid = percentileOf(features.ext_spectral_centroid);
  const persistence = percentileOf(features.dfa_alpha);
  const memory = percentileOf(features.ac_timescale);
  const vol = percentileOf(features.ext_volatility_ac1);
  const gain = Number(pred.predicted_best_qrc_advantage_vs_esn);
  if (Number.isFinite(gain) && gain > 0) {
    markers.push({ title: "Positive expected QRC gain", body: "The learned map predicts lower error for QRC than for ESN on this dataset." });
  }
  if (trend !== null && trend >= 0.75) {
    markers.push({ title: "Drifting or trend-like structure", body: "QRC-favorable atlas regions often involved slow changes in the level or state of the process." });
  }
  if (centroid !== null && centroid <= 0.30) {
    markers.push({ title: "Slow low-frequency structure", body: "Important information appears to live in slow components rather than only recent jumps." });
  }
  if (persistence !== null && persistence >= 0.65) {
    markers.push({ title: "Persistence / long memory", body: "The past appears to keep influencing the future over a longer horizon." });
  }
  if (memory !== null && memory >= 0.65) {
    markers.push({ title: "Long autocorrelation time", body: "The series changes slowly enough that memory traces may matter." });
  }
  if (vol !== null && vol >= 0.65) {
    markers.push({ title: "Volatility memory", body: "The size of fluctuations is itself temporally structured." });
  }
  if (support.ood_flag) {
    markers.push({ title: "Outside familiar atlas support", body: "This weakens the recommendation; direct benchmarking is needed." });
  }
  if (markers.length === 0) {
    markers.push({ title: "No strong QRC markers found", body: "The dataset does not clearly match the slow, stateful regimes where QRC was most useful." });
  }
  return markers;
}

function percentileOf(payload) {
  if (!payload) return null;
  const pct = Number(payload.atlas_percentile);
  return Number.isFinite(pct) ? Math.max(0, Math.min(1, pct)) : null;
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
  const labels = {
    ext_trend_strength: "Trend / drift strength",
    ext_spectral_centroid: "Speed of dominant signal",
    dfa_alpha: "Long-memory scaling",
    ac_timescale: "Autocorrelation memory",
    ext_volatility_ac1: "Volatility memory",
    snr_db: "Signal clarity"
  };
  return labels[name] || name.replace(/^ext_/, "").replaceAll("_", " ");
}

function featureMeaning(name, pct) {
  if (name === "ext_spectral_centroid") {
    return pct <= 0.30 ? "slow compared with atlas examples" : pct >= 0.70 ? "fast compared with atlas examples" : "typical speed";
  }
  if (name === "ext_trend_strength") {
    return pct >= 0.75 ? "strong drift marker" : pct <= 0.25 ? "weak drift marker" : "moderate drift marker";
  }
  if (name === "dfa_alpha") {
    return pct >= 0.65 ? "persistent / long-memory marker" : "not a strong persistence marker";
  }
  if (name === "ac_timescale") {
    return pct >= 0.65 ? "long memory trace" : "shorter memory trace";
  }
  if (name === "ext_volatility_ac1") {
    return pct >= 0.65 ? "fluctuation size has memory" : "weak volatility memory";
  }
  if (name === "snr_db") {
    return pct >= 0.70 ? "cleaner than many atlas examples" : pct <= 0.30 ? "noisier than many atlas examples" : "typical noise level";
  }
  return "position among atlas examples";
}

function percentileLabel(pct) {
  return `${(pct * 100).toFixed(0)}%`;
}

function formatFamilies(raw) {
  if (!raw) return "";
  return raw
    .split(";")
    .map(item => {
      const [family, weight] = item.split(":");
      const pct = Number(weight);
      const label = family.replaceAll("_", " ");
      return Number.isFinite(pct) ? `${label} (${(pct * 100).toFixed(0)}%)` : label;
    })
    .join(", ");
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
