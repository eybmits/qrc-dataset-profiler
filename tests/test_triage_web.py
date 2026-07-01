import numpy as np
import pandas as pd

from qrc_dataset_profiler.run_triage_web import _html_page, _parse_sep, _report_from_payload
from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS
from qrc_dataset_profiler.triage import fit_triage_model


def _fake_discovery(n=80):
    rng = np.random.default_rng(34)
    rows = []
    for i in range(n):
        trend = i / max(1, n - 1)
        row = {
            "dataset_id": f"d{i}:0:800",
            "name": f"d{i}",
            "family": "unit_root_trend" if trend > 0.7 else "chaotic_map",
            "source": "synthetic",
            "task_type": "forecast",
            "params": "{'generator': 'fake'}",
            "seed": i,
            "length": 800,
            "base_generator": "fake",
            "best_qrc_advantage_vs_esn": 0.2 if trend > 0.75 else -0.1,
        }
        for feature in FRONTIER_TIER_A_FIELDS:
            row[feature] = float(rng.normal(scale=0.1))
        row["ext_trend_strength"] = trend
        row["ext_spectral_centroid"] = 1.0 - trend
        rows.append(row)
    return pd.DataFrame(rows)


def test_html_page_contains_upload_form_and_api_route():
    html = _html_page()

    assert "QRC Regime Triage" in html
    assert "/api/triage" in html
    assert "fileInput" in html


def test_web_payload_returns_triage_report():
    discovery = _fake_discovery()
    model = fit_triage_model(discovery, seed=4)
    values = [str(0.01 * i + 0.05 * np.sin(i / 12)) for i in range(180)]
    payload = {
        "csv_text": "value\n" + "\n".join(values),
        "column": "value",
        "name": "web_test",
        "sep": ",",
        "header": True,
        "max_length": 120,
        "window": "tail",
    }

    report = _report_from_payload(payload, discovery_table=discovery, model=model)

    assert report["name"] == "web_test"
    assert report["dataset"]["used_length"] == 120
    assert "recommendation" in report["prediction"]
    assert "text_report" in report


def test_parse_sep_accepts_tab_alias():
    assert _parse_sep("tab") == "\t"
    assert _parse_sep("\\t") == "\t"
