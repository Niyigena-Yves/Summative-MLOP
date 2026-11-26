const CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"];
const NICE = { angular_leaf_spot: "Angular leaf spot", bean_rust: "Bean rust", healthy: "Healthy" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (c) => css(["--c1", "--c2", "--c3"][CLASSES.indexOf(c)]);
const $ = (id) => document.getElementById(id);
const charts = {};
let insightsLoaded = false;

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

// ---------- tabs ----------
function openTab(name) {
  const b = document.querySelector(`.tab[data-tab="${name}"]`);
  if (!b) return;
  document.querySelectorAll(".tab, .panel").forEach((e) => e.classList.remove("active"));
  b.classList.add("active");
  $(name).classList.add("active");
  history.replaceState(null, "", "#" + name);
  if (name === "insights" && !insightsLoaded) loadInsights();
  if (name === "retrain") loadRetrain();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => openTab(b.dataset.tab)));

// ---------- chart defaults ----------
function baseOpts(extra = {}) {
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--grid");
  Chart.defaults.font.family = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  return Object.assign({
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { display: false }, tooltip: { backgroundColor: css("--surface"), titleColor: css("--text"), bodyColor: css("--text"), borderColor: css("--border"), borderWidth: 1 } },
    scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: css("--grid") } } },
  }, extra);
}
function draw(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), config);
}
const barDs = (data) => ({ data, backgroundColor: CLASSES.map(color), borderRadius: 4, maxBarThickness: 56 });

// ---------- dashboard ----------
function fmtUptime(s) {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h ${m}m` : h ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`;
}

async function loadHealth() {
  try {
    const h = await api("/health");
    $("statusPill").textContent = h.model_loaded ? "Model online" : "Model not loaded";
    $("statusPill").className = "pill " + (h.model_loaded ? "ok" : "down");
    $("uptime").textContent = fmtUptime(h.uptime_seconds);
    $("container").textContent = "container " + h.container;
    $("version").textContent = h.model_version ? "v" + h.model_version : "–";
    $("trainedAt").textContent = h.model_trained_at ? "trained " + new Date(h.model_trained_at).toLocaleString() : "";
    $("served").textContent = h.predictions_served;
    $("avgLat").textContent = h.avg_prediction_latency_ms ? `avg ${h.avg_prediction_latency_ms} ms / prediction` : "no predictions yet";
    if (h.retraining) { $("statusPill").textContent = "Model online · retraining…"; }
  } catch (e) {
    $("statusPill").textContent = "API offline";
    $("statusPill").className = "pill down";
  }
  try { const u = await api("/uploads"); $("pending").textContent = u.pending_total; } catch (e) {}
}

async function loadModel() {
  let m;
  try { m = await api("/model"); } catch (e) { return; }
  const met = m.metrics;
  const tiles = [["Accuracy", met.accuracy], ["Precision (macro)", met.precision], ["Recall (macro)", met.recall], ["F1 score (macro)", met.f1], ["Log loss", met.loss]];
  $("metricTiles").innerHTML = tiles.map(([k, v]) =>
    `<div class="tile"><span class="k">${k}</span><span class="v">${k === "Log loss" ? v.toFixed(3) : (v * 100).toFixed(1) + "%"}</span><span class="h">${met.n_test} test images</span></div>`).join("");

  // confusion matrix as a heat grid (one hue, light -> dark)
  const cm = met.confusion_matrix, max = Math.max(...cm.flat());
  let html = `<div class="cm" style="grid-template-columns: 120px repeat(3, 1fr)"><div class="hd">true ↓ / pred →</div>`;
  html += CLASSES.map((c) => `<div class="hd">${NICE[c]}</div>`).join("");
  cm.forEach((row, i) => {
    html += `<div class="hd" style="text-align:left">${NICE[CLASSES[i]]}</div>`;
    row.forEach((v) => {
      const a = v / max;
      html += `<div title="${v} images" style="background: color-mix(in srgb, var(--c1) ${Math.round(a * 85 + 5)}%, transparent); color:${a > .5 ? "#fff" : "var(--text)"}">${v}</div>`;
    });
  });
  $("cm").innerHTML = html + "</div>";

  const pc = met.per_class;
  const metricNames = ["precision", "recall", "f1-score"];
  draw("perClassChart", {
    type: "bar",
    data: { labels: CLASSES.map((c) => NICE[c]), datasets: metricNames.map((k, i) => ({ label: k, data: CLASSES.map((c) => pc[c][k]), backgroundColor: css("--c1"), borderRadius: 4, maxBarThickness: 28, backgroundColor: [css("--c1"), css("--c2"), css("--c3")][i] })) },
    options: baseOpts({ plugins: { legend: { display: true, position: "top", labels: { boxWidth: 12 } } }, scales: { x: { grid: { display: false } }, y: { min: 0, max: 1, grid: { color: css("--grid") } } } }),
  });

  const hist = m.history;
  if (hist) {
    const ep = hist.accuracy.map((_, i) => i + 1);
    draw("historyChart", {
      type: "line",
      data: { labels: ep, datasets: [
        { label: "train accuracy", data: hist.accuracy, borderColor: css("--c1"), borderWidth: 2, pointRadius: 0, tension: .25 },
        { label: "validation accuracy", data: hist.val_accuracy, borderColor: css("--c2"), borderWidth: 2, pointRadius: 0, tension: .25 },
      ] },
      options: baseOpts({
        plugins: { legend: { display: true, position: "top", labels: { boxWidth: 12 } } },
        scales: { x: { title: { display: true, text: "epoch" }, grid: { display: false } }, y: { min: 0.3, max: 1, grid: { color: css("--grid") } } },
      }),
      plugins: [{ id: "phase", afterDraw(c) {
        const x = c.scales.x.getPixelForValue(hist.phase_boundary);
        const ctx = c.ctx; ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = css("--muted");
        ctx.beginPath(); ctx.moveTo(x, c.chartArea.top); ctx.lineTo(x, c.chartArea.bottom); ctx.stroke();
        ctx.fillStyle = css("--muted"); ctx.fillText("fine-tuning →", x + 4, c.chartArea.top + 12); ctx.restore();
      } }],
    });
  }
}

// ---------- predict ----------
let predictFile = null;
function setPredictFile(f) {
  if (!f) return;
  predictFile = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").hidden = false;
  $("dropText").textContent = f.name;
  $("predictBtn").disabled = false;
}
$("predictFile").addEventListener("change", (e) => setPredictFile(e.target.files[0]));
const drop = $("predictDrop");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); setPredictFile(e.dataTransfer.files[0]); });

$("predictBtn").addEventListener("click", async () => {
  const fd = new FormData(); fd.append("file", predictFile);
  $("predictBtn").disabled = true; $("predictBtn").textContent = "Predicting…";
  try {
    const r = await api("/predict", { method: "POST", body: fd });
    $("resultEmpty").hidden = true; $("result").hidden = false;
    $("resLabel").textContent = NICE[r.label];
    $("resSwatch").style.background = color(r.label);
    $("resConf").textContent = `Confidence ${(r.confidence * 100).toFixed(1)}%`;
    $("probBars").innerHTML = CLASSES.map((c) => {
      const p = r.probabilities[c] * 100;
      return `<div class="bar"><span>${NICE[c]}</span><div class="track"><div class="fill" style="width:${p}%;background:${color(c)}"></div></div><span class="pct">${p.toFixed(1)}%</span></div>`;
    }).join("");
    $("resAdvice").textContent = r.advice;
    $("resMeta").textContent = `model v${r.model_version} · ${r.latency_ms} ms · container ${r.container}`;
    loadHealth();
  } catch (e) {
    $("resultEmpty").hidden = false; $("result").hidden = true;
    $("resultEmpty").textContent = "Error: " + e.message;
  }
  $("predictBtn").disabled = false; $("predictBtn").textContent = "Predict";
});

// ---------- insights ----------
async function loadInsights() {
  const ins = await api("/insights");
  insightsLoaded = true;
  const f = ins.features, labels = CLASSES.map((c) => NICE[c]);
  $("legend").innerHTML = CLASSES.map((c) => `<span><i class="swatch" style="background:${color(c)}"></i>${NICE[c]}</span>`).join("");
  const simpleBar = (id, values, yTitle) => draw(id, { type: "bar", data: { labels, datasets: [barDs(values)] },
    options: baseOpts({ scales: { x: { grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: yTitle }, grid: { color: css("--grid") } } } }) });

  simpleBar("countChart", CLASSES.map((c) => ins.class_counts[c]), "images");
  simpleBar("greenChart", CLASSES.map((c) => f.greenness[c]), "mean G − (R+B)/2");
  simpleBar("lesionChart", CLASSES.map((c) => f.lesion_ratio[c] * 100), "% brown/yellow pixels");
  simpleBar("edgeChart", CLASSES.map((c) => f.edge_density[c]), "mean gradient");

  draw("rgbChart", { type: "bar",
    data: { labels: ["Red", "Green", "Blue"], datasets: CLASSES.map((c) => ({ label: NICE[c], data: [f.mean_r[c], f.mean_g[c], f.mean_b[c]], backgroundColor: color(c), borderRadius: 4, maxBarThickness: 30 })) },
    options: baseOpts({ plugins: { legend: { display: true, position: "top", labels: { boxWidth: 12 } } }, scales: { x: { grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: "mean intensity (0–255)" }, grid: { color: css("--grid") } } } }) });

  const hb = ins.histograms.brightness;
  const mids = hb.bins.slice(0, -1).map((b, i) => ((b + hb.bins[i + 1]) / 2).toFixed(0));
  draw("brightChart", { type: "line",
    data: { labels: mids, datasets: CLASSES.map((c) => ({ label: NICE[c], data: hb.counts[c], borderColor: color(c), borderWidth: 2, pointRadius: 0, tension: .3 })) },
    options: baseOpts({ plugins: { legend: { display: true, position: "top", labels: { boxWidth: 12 } } }, scales: { x: { title: { display: true, text: "image brightness (0–255)" }, grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: "images" }, grid: { color: css("--grid") } } } }) });

  const g = f.greenness, l = f.lesion_ratio, e = f.edge_density, b = f.brightness;
  const pct = (x) => (x * 100).toFixed(1) + "%";
  $("storyGreen").innerHTML = `<b>Story:</b> healthy leaves are the greenest (${g.healthy.toFixed(1)}), angular leaf spot is slightly lower (${g.angular_leaf_spot.toFixed(1)}) and bean rust drops sharply (${g.bean_rust.toFixed(1)}). Disease destroys chlorophyll; rust's dense reddish-brown pustules remove the most green, so greenness alone already flags many rust cases.`;
  $("storyLesion").innerHTML = `<b>Story:</b> diseased leaves carry far more brown/yellow tissue than healthy ones: angular leaf spot ${pct(l.angular_leaf_spot)}, bean rust ${pct(l.bean_rust)} vs healthy ${pct(l.healthy)}. Lesion share is the clearest sick-vs-healthy signal, which is why healthy leaves are almost never confused with diseased ones by the model.`;
  $("storyEdge").innerHTML = `<b>Story:</b> whole-image edge density is almost identical for all classes (${e.angular_leaf_spot.toFixed(1)}, ${e.bean_rust.toFixed(1)}, ${e.healthy.toFixed(1)}). Field photos contain leaf veins, soil and other leaves, so a global texture average hides the disease. The difference is <i>local</i> (tiny rust pustules vs larger angular patches), which is why a CNN with local convolution filters beats simple global features.`;
  $("storyRgb").innerHTML = `<b>Story:</b> the green channel tells the story: it falls from ${f.mean_g.healthy.toFixed(0)} (healthy) to ${f.mean_g.angular_leaf_spot.toFixed(0)} (angular leaf spot) and ${f.mean_g.bean_rust.toFixed(0)} (bean rust), while red stays flat or rises (${f.mean_r.healthy.toFixed(0)} → ${f.mean_r.angular_leaf_spot.toFixed(0)} / ${f.mean_r.bean_rust.toFixed(0)}). The red-to-green balance shifts toward the yellow/brown a farmer would see.`;
  $("storyBright").innerHTML = `<b>Story:</b> brightness overlaps heavily between classes (means ${b.angular_leaf_spot.toFixed(0)}, ${b.bean_rust.toFixed(0)}, ${b.healthy.toFixed(0)}); bean rust photos are only slightly darker. Photos were taken outdoors under varied light, so brightness is not a reliable cue, which is why we use contrast augmentation during training.`;
}

// ---------- upload & retrain ----------
$("uploadFiles").addEventListener("change", (e) => {
  const n = e.target.files.length;
  $("uploadText").textContent = n ? `${n} image(s) selected` : "Click to choose multiple images";
  $("uploadBtn").disabled = !n;
});
$("uploadBtn").addEventListener("click", async () => {
  const files = $("uploadFiles").files, fd = new FormData();
  fd.append("label", $("uploadLabel").value);
  for (const f of files) fd.append("files", f);
  $("uploadBtn").disabled = true; $("uploadMsg").className = "msg"; $("uploadMsg").textContent = "Uploading…";
  try {
    const r = await api("/upload", { method: "POST", body: fd });
    $("uploadMsg").textContent = `Saved ${r.saved} image(s) as "${r.label}" to the database.` +
      (r.rejected.length ? ` Rejected ${r.rejected.length}.` : "") +
      (r.auto_retrain_triggered ? " Threshold reached: automatic retraining started." : "");
    $("uploadFiles").value = ""; $("uploadText").textContent = "Click to choose multiple images";
    loadRetrain();
  } catch (e) { $("uploadMsg").className = "msg err"; $("uploadMsg").textContent = e.message; $("uploadBtn").disabled = false; }
});

$("retrainBtn").addEventListener("click", async () => {
  $("retrainMsg").className = "msg";
  try {
    const r = await api("/retrain", { method: "POST" });
    if (r.started) $("retrainMsg").textContent = `Retraining run #${r.run_id} started on ${r.n_samples} new image(s)…`;
    else { $("retrainMsg").className = "msg err"; $("retrainMsg").textContent = r.reason; }
    loadRetrain();
  } catch (e) { $("retrainMsg").className = "msg err"; $("retrainMsg").textContent = e.message; }
});

let pollTimer = null;
async function loadRetrain() {
  const s = await api("/retrain/status");
  const u = await api("/uploads");
  $("threshold").textContent = s.threshold;
  $("pendingBox").innerHTML = CLASSES.map((c) => `<div><b>${u.pending[c] || 0}</b>${NICE[c]}</div>`).join("") + `<div><b>${u.pending_total}</b>pending total</div>`;
  $("retrainBtn").disabled = s.running;
  $("retrainBtn").textContent = s.running ? "Retraining in progress…" : "Retrain model";
  $("runsTable").querySelector("tbody").innerHTML = s.runs.map((r) => {
    const res = r.result || {}, m = res.new_metrics || {};
    return `<tr><td>${r.id}</td><td>${r.trigger}</td><td>${r.status}${r.error ? " – " + r.error : ""}</td><td>${new Date(r.started_at).toLocaleString()}</td><td>${r.n_samples}</td>
      <td>${m.accuracy != null ? (m.accuracy * 100).toFixed(1) + "%" : "–"}</td><td>${m.f1 != null ? (m.f1 * 100).toFixed(1) + "%" : "–"}</td>
      <td>${res.promoted == null ? "–" : res.promoted ? "yes → v" + res.version : "no (kept v" + res.version + ")"}</td><td>${res.duration_sec ? res.duration_sec + " s" : "–"}</td></tr>`;
  }).join("") || `<tr><td colspan="9" class="muted">No retraining runs yet.</td></tr>`;
  clearTimeout(pollTimer);
  if (s.running) pollTimer = setTimeout(loadRetrain, 4000);
  else if (s.runs.length) { loadModel(); loadHealth(); }
}

loadHealth(); loadModel();
if (location.hash) openTab(location.hash.slice(1));
setInterval(loadHealth, 5000);
