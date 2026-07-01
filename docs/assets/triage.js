const MODEL_URL = "assets/triage_model.json";
const MAX_ANALYSIS_LENGTH = 800;

let triageModel = null;

document.addEventListener("DOMContentLoaded", () => {
  setupTriageUi().catch((error) => {
    showAnalyzerMessage(`Could not load the browser model: ${error.message}`, "error");
  });
});

async function setupTriageUi() {
  const response = await fetch(MODEL_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  triageModel = await response.json();

  const fileInput = document.getElementById("csvFile");
  const textArea = document.getElementById("csvText");
  const analyzeButton = document.getElementById("analyzeButton");
  const demoButton = document.getElementById("demoButton");

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    textArea.value = await file.text();
    showAnalyzerMessage(`Loaded ${file.name}. Press Analyze dataset.`, "info");
  });

  demoButton.addEventListener("click", () => {
    textArea.value = makeDemoCsv();
    document.getElementById("columnName").value = "value";
    showAnalyzerMessage("Loaded a synthetic drifting example.", "info");
  });

  analyzeButton.addEventListener("click", () => {
    try {
      const result = analyzeCsv(textArea.value, document.getElementById("columnName").value);
      renderResult(result);
    } catch (error) {
      showAnalyzerMessage(error.message, "error");
    }
  });

  showAnalyzerMessage("Upload or paste a CSV to get a browser-side QRC triage result.", "info");
}

function analyzeCsv(csvText, columnSpec) {
  if (!triageModel) {
    throw new Error("The browser model is still loading.");
  }
  const parsed = parseSeriesFromCsv(csvText, columnSpec);
  const windowed = takeTail(finiteFill(parsed.values), MAX_ANALYSIS_LENGTH);
  if (windowed.length < 32) {
    throw new Error("Please provide at least 32 numeric samples. A few hundred are better.");
  }
  const features = computeFeatures(windowed, parsed.rawLength, parsed.missingFraction);
  const modelInput = triageModel.features.map((feature) => {
    const value = features[feature];
    return Number.isFinite(value) ? value : triageModel.imputer_medians[feature] || 0;
  });
  const probability = sigmoid(evaluateEnsemble(triageModel.classifier, modelInput));
  const predictedAdvantage = evaluateEnsemble(triageModel.regressor, modelInput);
  const percentiles = {};
  for (const feature of triageModel.features) {
    percentiles[feature] = percentileFromQuantiles(feature, features[feature]);
  }
  const support = supportSummary(percentiles);
  const markers = markerSummary(percentiles);
  const recommendation = recommendationFromScore(probability, predictedAdvantage, support, markers);
  return {
    parsed,
    features,
    percentiles,
    probability,
    predictedAdvantage,
    support,
    markers,
    recommendation,
    validation: triageModel.validation_metrics,
  };
}

function parseSeriesFromCsv(text, columnSpec) {
  const clean = String(text || "").trim();
  if (!clean) {
    throw new Error("Paste a CSV or choose a file first.");
  }
  const delimiter = detectDelimiter(clean);
  const rows = parseDelimited(clean, delimiter).filter((row) => row.some((cell) => String(cell).trim() !== ""));
  if (rows.length < 2) {
    throw new Error("The CSV needs at least two rows.");
  }
  const first = rows[0];
  const firstNumeric = first.filter((cell) => Number.isFinite(parseNumber(cell))).length;
  const hasHeader = firstNumeric < Math.max(1, Math.ceil(first.length / 2));
  const headers = hasHeader ? first.map((cell, index) => String(cell || `column_${index}`).trim()) : first.map((_, index) => `column_${index}`);
  const dataRows = hasHeader ? rows.slice(1) : rows;
  const columnIndex = chooseColumn(headers, dataRows, columnSpec);
  const rawValues = dataRows.map((row) => parseNumber(row[columnIndex]));
  const finiteCount = rawValues.filter(Number.isFinite).length;
  if (finiteCount < 32) {
    throw new Error(`Column "${headers[columnIndex]}" has only ${finiteCount} numeric values.`);
  }
  return {
    values: rawValues,
    column: headers[columnIndex],
    delimiter: delimiter === "\t" ? "tab" : delimiter,
    hasHeader,
    rawLength: rawValues.length,
    finiteCount,
    missingFraction: 1 - finiteCount / Math.max(1, rawValues.length),
  };
}

function detectDelimiter(text) {
  const sample = text.split(/\r?\n/).slice(0, 12).join("\n");
  const candidates = [",", ";", "\t"];
  let best = ",";
  let bestCount = -1;
  for (const delimiter of candidates) {
    const count = Array.from(sample).filter((char) => char === delimiter).length;
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return bestCount <= 0 ? "," : best;
}

function parseDelimited(text, delimiter) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === delimiter && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  row.push(cell);
  rows.push(row);
  return rows;
}

function chooseColumn(headers, rows, columnSpec) {
  const requested = String(columnSpec || "").trim();
  if (requested) {
    const exact = headers.findIndex((header) => header === requested);
    if (exact >= 0) return exact;
    const lower = headers.findIndex((header) => header.toLowerCase() === requested.toLowerCase());
    if (lower >= 0) return lower;
    const numericIndex = Number.parseInt(requested, 10);
    if (Number.isInteger(numericIndex) && numericIndex >= 0 && numericIndex < headers.length) return numericIndex;
    throw new Error(`Could not find column "${requested}". Available columns: ${headers.join(", ")}`);
  }

  const priorities = ["value", "target", "y", "series", "signal", "close", "load", "price"];
  const numericCounts = headers.map((_, index) => rows.filter((row) => Number.isFinite(parseNumber(row[index]))).length);
  for (const name of priorities) {
    const index = headers.findIndex((header) => header.toLowerCase() === name);
    if (index >= 0 && numericCounts[index] >= 32) return index;
  }
  let best = 0;
  for (let i = 1; i < numericCounts.length; i += 1) {
    if (numericCounts[i] > numericCounts[best]) best = i;
  }
  return best;
}

function parseNumber(value) {
  if (value === undefined || value === null) return NaN;
  let text = String(value).trim();
  if (!text) return NaN;
  text = text.replace(/\s/g, "");
  if (/^-?\d+,\d+$/.test(text)) text = text.replace(",", ".");
  const number = Number(text);
  return Number.isFinite(number) ? number : NaN;
}

function computeFeatures(values, rawLength, missingFraction) {
  const raw = values.slice();
  const z = zscore(raw);
  const spectral = spectralFeatures(z);
  const sampleEntropyValue = sampleEntropy(z, 2);
  return {
    length: raw.length,
    triage_raw_length: rawLength,
    triage_used_length: raw.length,
    missing_frac: missingFraction,
    ac_timescale: acTimescale(z),
    dfa_alpha: dfaAlpha(z),
    spectral_entropy: spectral.spectral_entropy,
    dom_freq: spectral.dom_freq,
    spectral_flatness: spectral.spectral_flatness,
    perm_entropy: permutationEntropy(z, 4),
    sample_entropy: sampleEntropyValue,
    hurst_rs: hurstRs(z),
    forecastability: Number.isFinite(spectral.spectral_entropy) ? 1 - spectral.spectral_entropy : NaN,
    ext_volatility_ac1: volatilityAc1(z),
    ext_arch_lm5: archLm(z, 5),
    ext_psd_slope: spectral.ext_psd_slope,
    ext_spectral_centroid: spectral.ext_spectral_centroid,
    ext_trend_strength: trendStrength(raw),
    ext_changepoint_count: changepointCount(raw),
    ext_lz_complexity: lzComplexity(z),
  };
}

function finiteFill(values) {
  const arr = values.map((value) => (Number.isFinite(value) ? value : NaN));
  let first = -1;
  for (let i = 0; i < arr.length; i += 1) {
    if (Number.isFinite(arr[i])) {
      first = i;
      break;
    }
  }
  if (first < 0) return [];
  for (let i = 0; i < first; i += 1) arr[i] = arr[first];
  let left = first;
  for (let right = first + 1; right < arr.length; right += 1) {
    if (!Number.isFinite(arr[right])) continue;
    for (let i = left + 1; i < right; i += 1) {
      const weight = (i - left) / (right - left);
      arr[i] = arr[left] * (1 - weight) + arr[right] * weight;
    }
    left = right;
  }
  for (let i = left + 1; i < arr.length; i += 1) arr[i] = arr[left];
  return arr;
}

function takeTail(values, maxLength) {
  return values.length > maxLength ? values.slice(values.length - maxLength) : values.slice();
}

function mean(values) {
  return values.reduce((acc, value) => acc + value, 0) / Math.max(1, values.length);
}

function variance(values) {
  const mu = mean(values);
  return mean(values.map((value) => (value - mu) ** 2));
}

function std(values) {
  return Math.sqrt(Math.max(0, variance(values)));
}

function zscore(values) {
  const mu = mean(values);
  const sd = std(values);
  if (!Number.isFinite(sd) || sd < 1e-12) return values.map(() => 0);
  return values.map((value) => (value - mu) / sd);
}

function acTimescale(values) {
  const n = values.length;
  if (n < 4) return NaN;
  const maxLag = Math.min(500, Math.floor(n / 3));
  const denom = values.reduce((acc, value) => acc + value * value, 0) || 1e-12;
  for (let lag = 1; lag <= maxLag; lag += 1) {
    let c = 0;
    for (let i = 0; i < n - lag; i += 1) c += values[i] * values[i + lag];
    if (Math.abs(c / denom) < 1 / Math.E) return lag;
  }
  return maxLag;
}

function trendStrength(values) {
  const n = values.length;
  if (n < 16) return NaN;
  const t = values.map((_, i) => -1 + (2 * i) / Math.max(1, n - 1));
  const tMean = mean(t);
  const yMean = mean(values);
  let cov = 0;
  let vt = 0;
  for (let i = 0; i < n; i += 1) {
    cov += (t[i] - tMean) * (values[i] - yMean);
    vt += (t[i] - tMean) ** 2;
  }
  const slope = cov / Math.max(vt, 1e-12);
  const intercept = yMean - slope * tMean;
  const residuals = values.map((value, i) => value - (intercept + slope * t[i]));
  const denom = variance(values);
  if (denom <= 1e-12) return 0;
  return clamp(1 - variance(residuals) / denom, 0, 1);
}

function volatilityAc1(values) {
  if (values.length < 4) return NaN;
  const v = [];
  for (let i = 1; i < values.length; i += 1) v.push(Math.abs(values[i] - values[i - 1]));
  return correlation(v.slice(0, -1), v.slice(1));
}

function correlation(a, b) {
  if (a.length !== b.length || a.length < 2) return NaN;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    num += da * db;
    va += da * da;
    vb += db * db;
  }
  const denom = Math.sqrt(va * vb);
  return denom <= 1e-12 ? 0 : num / denom;
}

function spectralFeatures(values) {
  const x = resample(values, Math.min(256, values.length));
  const n = x.length;
  if (n < 16) {
    return {
      spectral_entropy: NaN,
      dom_freq: NaN,
      spectral_flatness: NaN,
      ext_psd_slope: NaN,
      ext_spectral_centroid: NaN,
    };
  }
  const freqs = [];
  const psd = [];
  for (let k = 1; k <= Math.floor(n / 2); k += 1) {
    let re = 0;
    let im = 0;
    for (let j = 0; j < n; j += 1) {
      const angle = (-2 * Math.PI * k * j) / n;
      re += x[j] * Math.cos(angle);
      im += x[j] * Math.sin(angle);
    }
    freqs.push(k / n);
    psd.push(Math.max(re * re + im * im, 1e-15));
  }
  const total = psd.reduce((acc, value) => acc + value, 0);
  const weights = psd.map((value) => value / Math.max(total, 1e-15));
  const entropy = -weights.reduce((acc, value) => acc + value * Math.log(value + 1e-15), 0) / Math.log(weights.length);
  const centroid = weights.reduce((acc, value, index) => acc + value * freqs[index], 0);
  let maxIndex = 0;
  for (let i = 1; i < psd.length; i += 1) if (psd[i] > psd[maxIndex]) maxIndex = i;
  const flatness = Math.exp(mean(psd.map((value) => Math.log(value)))) / Math.max(mean(psd), 1e-15);
  const slope = linearSlope(freqs.map(Math.log), psd.map(Math.log));
  return {
    spectral_entropy: entropy,
    dom_freq: freqs[maxIndex],
    spectral_flatness: flatness,
    ext_psd_slope: slope,
    ext_spectral_centroid: centroid,
  };
}

function resample(values, targetLength) {
  if (values.length <= targetLength) return values.slice();
  const out = [];
  for (let i = 0; i < targetLength; i += 1) {
    const pos = (i * (values.length - 1)) / Math.max(1, targetLength - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(values.length - 1, lo + 1);
    const w = pos - lo;
    out.push(values[lo] * (1 - w) + values[hi] * w);
  }
  return out;
}

function dfaAlpha(values) {
  const n = values.length;
  if (n < 96) return NaN;
  const mu = mean(values);
  const y = [];
  values.reduce((acc, value) => {
    const next = acc + value - mu;
    y.push(next);
    return next;
  }, 0);
  const sizes = uniqueLogSizes(8, Math.max(16, Math.floor(n / 4)), 12);
  const used = [];
  const flucts = [];
  for (const size of sizes) {
    const segments = Math.floor(n / size);
    if (segments < 4) continue;
    const rms = [];
    for (let s = 0; s < segments; s += 1) {
      const start = s * size;
      const seg = y.slice(start, start + size);
      const t = seg.map((_, i) => i);
      const slope = linearSlope(t, seg);
      const intercept = mean(seg) - slope * mean(t);
      const detrended = seg.map((value, i) => value - (intercept + slope * i));
      rms.push(Math.sqrt(mean(detrended.map((value) => value * value))));
    }
    const val = Math.sqrt(mean(rms.map((value) => value * value)));
    if (val > 0 && Number.isFinite(val)) {
      used.push(size);
      flucts.push(val);
    }
  }
  return used.length < 5 ? NaN : linearSlope(used.map(Math.log), flucts.map(Math.log));
}

function hurstRs(values) {
  const n = values.length;
  if (n < 128) return NaN;
  const sizes = uniqueLogSizes(16, Math.max(32, Math.floor(n / 4)), 10);
  const used = [];
  const rs = [];
  for (const size of sizes) {
    const segments = Math.floor(n / size);
    if (segments < 3) continue;
    const vals = [];
    for (let s = 0; s < segments; s += 1) {
      const seg = values.slice(s * size, s * size + size);
      const mu = mean(seg);
      let c = 0;
      const cumulative = seg.map((value) => {
        c += value - mu;
        return c;
      });
      const range = Math.max(...cumulative) - Math.min(...cumulative);
      const sd = std(seg);
      if (sd > 1e-12 && range > 0) vals.push(range / sd);
    }
    if (vals.length) {
      used.push(size);
      rs.push(mean(vals));
    }
  }
  return used.length < 4 ? NaN : linearSlope(used.map(Math.log), rs.map(Math.log));
}

function permutationEntropy(values, order) {
  const n = values.length - order + 1;
  if (n <= order) return NaN;
  const counts = new Map();
  for (let i = 0; i < n; i += 1) {
    const window = values.slice(i, i + order);
    const pattern = window.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value || a.index - b.index).map((item) => item.index).join("-");
    counts.set(pattern, (counts.get(pattern) || 0) + 1);
  }
  const probs = Array.from(counts.values()).map((count) => count / n);
  const entropy = -probs.reduce((acc, p) => acc + p * Math.log(p + 1e-15), 0);
  return entropy / Math.log(factorial(order));
}

function sampleEntropy(values, order) {
  let x = values;
  if (x.length > 260) x = resample(x, 260);
  const tolerance = 0.2 * std(x);
  if (x.length < order + 3 || tolerance <= 0) return NaN;
  const countMatches = (m) => {
    const count = x.length - m + 1;
    let matches = 0;
    for (let i = 0; i < count - 1; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        let dist = 0;
        for (let k = 0; k < m; k += 1) dist = Math.max(dist, Math.abs(x[i + k] - x[j + k]));
        if (dist <= tolerance) matches += 1;
      }
    }
    return matches;
  };
  const a = countMatches(order + 1);
  const b = countMatches(order);
  return a <= 0 || b <= 0 ? NaN : -Math.log(a / b);
}

function lzComplexity(values) {
  if (values.length < 4) return NaN;
  const med = median(values);
  const bits = values.map((value) => (value > med ? "1" : "0")).join("");
  const seen = new Set();
  let count = 0;
  let i = 0;
  while (i < bits.length) {
    let j = i + 1;
    while (j <= bits.length && seen.has(bits.slice(i, j))) j += 1;
    seen.add(bits.slice(i, j));
    count += 1;
    i = j;
  }
  const n = Math.max(bits.length, 2);
  return (count * Math.log2(n)) / n;
}

function archLm(values, lags) {
  const e2 = values.map((value) => value * value);
  if (e2.length <= lags + 20) return NaN;
  const y = e2.slice(lags);
  const X = [];
  for (let i = lags; i < e2.length; i += 1) {
    const row = [1];
    for (let lag = 1; lag <= lags; lag += 1) row.push(e2[i - lag]);
    X.push(row);
  }
  const coef = leastSquares(X, y);
  if (!coef) return NaN;
  const pred = X.map((row) => row.reduce((acc, value, index) => acc + value * coef[index], 0));
  const yMean = mean(y);
  const ssTot = y.reduce((acc, value) => acc + (value - yMean) ** 2, 0);
  const ssErr = y.reduce((acc, value, index) => acc + (value - pred[index]) ** 2, 0);
  if (ssTot <= 1e-12) return 0;
  return Math.max(0, y.length * (1 - ssErr / ssTot));
}

function changepointCount(values) {
  const n = values.length;
  if (n < 80) return 0;
  const window = Math.max(20, Math.floor(n / 40));
  const smooth = movingAverage(values, window);
  const diff = [];
  for (let i = window; i < smooth.length; i += 1) diff.push(Math.abs(smooth[i] - smooth[i - window]));
  if (!diff.length) return 0;
  const med = median(diff);
  const mad = median(diff.map((value) => Math.abs(value - med)));
  const threshold = Math.max(med + 4 * mad, 1e-12);
  let count = 0;
  let last = -window;
  for (let i = 1; i < diff.length - 1; i += 1) {
    if (diff[i] >= threshold && diff[i] >= diff[i - 1] && diff[i] >= diff[i + 1] && i - last >= window) {
      count += 1;
      last = i;
    }
  }
  return count;
}

function movingAverage(values, window) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(window, i + 1));
  }
  return out;
}

function uniqueLogSizes(lo, hi, count) {
  const values = [];
  for (let i = 0; i < count; i += 1) {
    const t = count === 1 ? 0 : i / (count - 1);
    values.push(Math.round(Math.exp(Math.log(lo) * (1 - t) + Math.log(hi) * t)));
  }
  return Array.from(new Set(values)).filter((value) => value >= lo && value <= hi);
}

function linearSlope(x, y) {
  if (x.length !== y.length || x.length < 2) return NaN;
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let den = 0;
  for (let i = 0; i < x.length; i += 1) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  return den <= 1e-12 ? 0 : num / den;
}

function leastSquares(X, y) {
  const p = X[0].length;
  const A = Array.from({ length: p }, () => Array(p).fill(0));
  const b = Array(p).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < p; j += 1) {
      b[j] += X[i][j] * y[i];
      for (let k = 0; k < p; k += 1) A[j][k] += X[i][j] * X[i][k];
    }
  }
  return solveLinearSystem(A, b);
}

function solveLinearSystem(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let i = 0; i < n; i += 1) {
    let pivot = i;
    for (let r = i + 1; r < n; r += 1) if (Math.abs(M[r][i]) > Math.abs(M[pivot][i])) pivot = r;
    if (Math.abs(M[pivot][i]) < 1e-12) return null;
    [M[i], M[pivot]] = [M[pivot], M[i]];
    const scale = M[i][i];
    for (let c = i; c <= n; c += 1) M[i][c] /= scale;
    for (let r = 0; r < n; r += 1) {
      if (r === i) continue;
      const factor = M[r][i];
      for (let c = i; c <= n; c += 1) M[r][c] -= factor * M[i][c];
    }
  }
  return M.map((row) => row[n]);
}

function evaluateEnsemble(ensemble, x) {
  let raw = ensemble.init_log_odds !== undefined ? ensemble.init_log_odds : ensemble.init_value;
  for (const tree of ensemble.trees) raw += ensemble.learning_rate * evaluateTree(tree, x);
  return raw;
}

function evaluateTree(tree, x) {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    const featureIndex = tree.feature[node];
    const threshold = tree.threshold[node];
    node = x[featureIndex] <= threshold ? tree.children_left[node] : tree.children_right[node];
  }
  return tree.value[node];
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function percentileFromQuantiles(feature, value) {
  if (!Number.isFinite(value)) return null;
  const qs = triageModel.feature_quantiles[feature];
  const grid = triageModel.quantile_grid;
  if (!qs || !grid) return null;
  if (value <= qs[0]) return 0;
  if (value >= qs[qs.length - 1]) return 1;
  for (let i = 0; i < qs.length - 1; i += 1) {
    if (value >= qs[i] && value <= qs[i + 1]) {
      const span = qs[i + 1] - qs[i];
      const local = span <= 1e-12 ? 0 : (value - qs[i]) / span;
      return clamp(grid[i] + local * (grid[i + 1] - grid[i]), 0, 1);
    }
  }
  return null;
}

function supportSummary(percentiles) {
  const values = Object.values(percentiles).filter((value) => value !== null);
  const inside = values.filter((value) => value > 0.02 && value < 0.98).length;
  const extreme = values.filter((value) => value <= 0.005 || value >= 0.995).length;
  const support = values.length ? inside / values.length : 0;
  return {
    score: support,
    outside: support < 0.72 || extreme >= 4,
    extreme,
  };
}

function markerSummary(percentiles) {
  const pct = (name) => percentiles[name];
  const slow = (pct("ext_spectral_centroid") !== null && pct("ext_spectral_centroid") <= 0.35)
    || (pct("dom_freq") !== null && pct("dom_freq") <= 0.35)
    || (pct("ext_psd_slope") !== null && pct("ext_psd_slope") <= 0.30);
  const drift = pct("ext_trend_strength") !== null && pct("ext_trend_strength") >= 0.72;
  const memory = (pct("dfa_alpha") !== null && pct("dfa_alpha") >= 0.62)
    || (pct("ac_timescale") !== null && pct("ac_timescale") >= 0.62)
    || (pct("hurst_rs") !== null && pct("hurst_rs") >= 0.62);
  const volatility = (pct("ext_volatility_ac1") !== null && pct("ext_volatility_ac1") >= 0.62)
    || (pct("ext_arch_lm5") !== null && pct("ext_arch_lm5") >= 0.62);
  const moderateComplexity = pct("perm_entropy") !== null && pct("perm_entropy") >= 0.25 && pct("perm_entropy") <= 0.85;
  return { slow, drift, memory, volatility, moderateComplexity };
}

function recommendationFromScore(probability, advantage, support, markers) {
  if (support.outside) {
    return {
      level: "ood",
      title: "Run a direct benchmark.",
      badge: "Different from atlas",
      body: "This time series is not close enough to the browser atlas markers. Do not trust a screening score alone.",
    };
  }
  const statefulCount = [markers.slow, markers.drift, markers.memory, markers.volatility].filter(Boolean).length;
  if ((probability >= 0.45 && advantage >= 0.02) || (probability >= 0.35 && statefulCount >= 3 && advantage > -0.01)) {
    return {
      level: "high",
      title: "QRC is worth testing.",
      badge: "QRC priority",
      body: "The series has the slow, stateful markers that were most often associated with QRC-useful cases.",
    };
  }
  if (probability >= 0.22 || advantage > 0) {
    return {
      level: "worth",
      title: "QRC may be worth testing.",
      badge: "Possible QRC case",
      body: "Some markers point toward a QRC-favorable regime, but ESN remains a strong default.",
    };
  }
  return {
    level: "low",
    title: "ESN is probably enough first.",
    badge: "ESN first",
    body: "The series does not strongly match the slow, stateful regimes where QRC was most useful in the atlas.",
  };
}

function renderResult(result) {
  const panel = document.getElementById("resultPanel");
  const rec = result.recommendation;
  panel.hidden = false;
  panel.className = `result ${rec.level}`;
  panel.innerHTML = `
    <div class="result-head">
      <span class="result-badge">${escapeHtml(rec.badge)}</span>
      <span>${escapeHtml(result.parsed.finiteCount)} numeric samples found · ${escapeHtml(result.features.triage_used_length)} analyzed · column "${escapeHtml(result.parsed.column)}"</span>
    </div>
    <h3>${escapeHtml(rec.title)}</h3>
    <p>${escapeHtml(rec.body)}</p>
    <div class="score-row">
      <div><strong>${formatPct(result.probability)}</strong><span>QRC testing priority</span></div>
      <div><strong>${formatSigned(result.predictedAdvantage)}</strong><span>estimated QRC advantage</span></div>
      <div><strong>${formatPct(result.support.score)}</strong><span>similarity to atlas markers</span></div>
    </div>
    <div class="marker-grid">
      ${markerChip("Slow structure", result.markers.slow)}
      ${markerChip("Drift or trend", result.markers.drift)}
      ${markerChip("Long memory", result.markers.memory)}
      ${markerChip("Volatility memory", result.markers.volatility)}
    </div>
    <details>
      <summary>Show measured markers</summary>
      ${featureTable(result)}
    </details>
    <p class="boundary">This browser result is a fast screening signal. It does not prove quantum advantage and does not run QRC or ESN on your data.</p>
  `;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function markerChip(label, active) {
  return `<div class="marker ${active ? "active" : ""}"><strong>${active ? "Yes" : "No"}</strong><span>${escapeHtml(label)}</span></div>`;
}

function featureTable(result) {
  const labels = {
    ext_trend_strength: "Trend / drift",
    ext_spectral_centroid: "Signal speed",
    ac_timescale: "Autocorrelation memory",
    dfa_alpha: "Long-memory scaling",
    ext_volatility_ac1: "Volatility memory",
    spectral_entropy: "Spectral entropy",
    perm_entropy: "Pattern entropy",
    ext_changepoint_count: "Regime-change marker",
  };
  return `<table>
    <thead><tr><th>Marker</th><th>Value</th><th>Atlas position</th></tr></thead>
    <tbody>
      ${Object.entries(labels).map(([feature, label]) => `
        <tr>
          <td>${escapeHtml(label)}</td>
          <td>${formatNumber(result.features[feature])}</td>
          <td>${result.percentiles[feature] === null ? "-" : formatPct(result.percentiles[feature])}</td>
        </tr>
      `).join("")}
    </tbody>
  </table>`;
}

function showAnalyzerMessage(message, level) {
  const panel = document.getElementById("resultPanel");
  if (!panel) return;
  panel.hidden = false;
  panel.className = `result ${level === "error" ? "ood" : "info"}`;
  panel.innerHTML = `<p>${escapeHtml(message)}</p>`;
}

function makeDemoCsv() {
  const lines = ["value"];
  let level = 0;
  let volatility = 0.25;
  for (let i = 0; i < 420; i += 1) {
    if (i === 160) volatility = 0.55;
    if (i === 285) volatility = 0.18;
    level += 0.015 + 0.012 * Math.sin(i / 80);
    const slow = 1.2 * Math.sin(i / 55) + 0.6 * Math.sin(i / 130);
    const noise = volatility * (Math.sin(i * 1.7) + 0.5 * Math.sin(i * 0.37));
    lines.push((level + slow + noise).toFixed(6));
  }
  return lines.join("\n");
}

function median(values) {
  const x = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!x.length) return NaN;
  const mid = Math.floor(x.length / 2);
  return x.length % 2 ? x[mid] : 0.5 * (x[mid - 1] + x[mid]);
}

function factorial(n) {
  let out = 1;
  for (let i = 2; i <= n; i += 1) out *= i;
  return out;
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function formatPct(value) {
  return Number.isFinite(value) ? `${Math.round(100 * value)}%` : "-";
}

function formatSigned(value) {
  return Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(3)}` : "-";
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
