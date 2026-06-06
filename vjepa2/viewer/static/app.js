const state = {
  runs: [],
  runId: null,
  manifest: null,
  inventory: null,
  examples: [],
  selectedExampleId: null,
  popSort: "absgap",
};

const els = {
  runMeta: document.querySelector("#runMeta"),
  runSelect: document.querySelector("#runSelect"),
  exampleFilter: document.querySelector("#exampleFilter"),
  metricSort: document.querySelector("#metricSort"),
  exampleList: document.querySelector("#exampleList"),
  exampleDetail: document.querySelector("#exampleDetail"),
  homeStats: document.querySelector("#homeStats"),
  populationBody: document.querySelector("#populationBody"),
  populationEmpty: document.querySelector("#populationEmpty"),
  inventoryFilter: document.querySelector("#inventoryFilter"),
  inventoryView: document.querySelector("#inventoryView"),
  inventoryBody: document.querySelector("#inventoryBody"),
  inventoryEmpty: document.querySelector("#inventoryEmpty"),
};

// When built for static hosting (GitHub Pages), the built index.html sets
// window.VIEWER_STATIC to the data directory; the viewer then reads pre-baked JSON
// instead of the serve.py API, and curation save is disabled (read-only site).
const STATIC_BASE = (typeof window !== "undefined" && window.VIEWER_STATIC) || null;

function assetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//.test(path) || path.startsWith("/")) return path;
  const enc = path.split("/").map(encodeURIComponent).join("/");
  if (STATIC_BASE) return `${STATIC_BASE}/${encodeURIComponent(state.runId)}/${enc}`;
  return `/assets/${encodeURIComponent(state.runId)}/${enc}`;
}

function formatValue(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(5);
  if (Array.isArray(value)) return value.join(", ");
  if (value === null || value === undefined) return "";
  return String(value);
}

function blockOf(example) {
  // id like "O1:15_p4" -> block "O1"; falls back gracefully.
  return String(example.id || "").split(":")[0] || "?";
}

function metricKeys() {
  const keys = new Set();
  for (const example of state.examples) {
    Object.keys(example.metrics || {}).forEach((key) => keys.add(key));
  }
  return [...keys].sort();
}

function scoreFor(example, key) {
  if (key === "id") return example.id || "";
  if (key === "label") return example.label || "";
  return example.metrics?.[key];
}

function sortedExamples() {
  const filter = els.exampleFilter.value.trim().toLowerCase();
  const sort = els.metricSort.value;
  return [...state.examples]
    .filter((example) => {
      const haystack = [example.id, example.label, JSON.stringify(example.metrics || {})].join(" ").toLowerCase();
      return haystack.includes(filter);
    })
    .sort((a, b) => {
      const av = scoreFor(a, sort);
      const bv = scoreFor(b, sort);
      if (typeof av === "number" && typeof bv === "number") return bv - av;
      return String(av ?? "").localeCompare(String(bv ?? ""));
    });
}

function renderRunMeta() {
  const run = state.manifest?.run || {};
  const pieces = [run.id, run.created, run.command].filter(Boolean);
  els.runMeta.textContent = pieces.length ? pieces.join(" | ") : "Manifest loaded.";
}

function renderSortControls() {
  const keys = metricKeys();
  const options = ["id", "label", ...keys];
  els.metricSort.innerHTML = options.map((key) => `<option value="${key}">${key}</option>`).join("");
  const defaultMetric = keys.find((key) => /gap|error|surprise|drift|loss/i.test(key)) || options[0];
  els.metricSort.value = defaultMetric;
}

function renderExampleList() {
  const items = sortedExamples();
  if (!state.selectedExampleId && items.length) state.selectedExampleId = items[0].id;
  els.exampleList.innerHTML = items
    .map((example) => {
      const selected = example.id === state.selectedExampleId ? " is-active" : "";
      const metric = els.metricSort.value;
      const value = scoreFor(example, metric);
      const meta = [example.label, value !== undefined ? `${metric}: ${formatValue(value)}` : ""].filter(Boolean).join(" | ");
      return `<button class="${selected}" data-example="${example.id}">
        <span class="itemTitle">${example.id}</span>
        <span class="itemMeta">${meta || "No metrics"}</span>
      </button>`;
    })
    .join("");
}

function metricGrid(metrics = {}) {
  const entries = Object.entries(metrics);
  if (!entries.length) return `<div class="empty">No metrics for this example.</div>`;
  return `<div class="metricGrid">${entries
    .map(([key, value]) => `<div class="metric"><span>${key}</span><strong>${formatValue(value)}</strong></div>`)
    .join("")}</div>`;
}

function inventoryItemFor(spec) {
  return (state.inventory?.items || []).find((item) => item.spec === spec);
}

function statusLabel(value) {
  return value === "unreviewed" ? "needs review" : (value || "");
}

// --- Examples tab: side-by-side possible/impossible player with a playhead riding the surprise curve ---

let rescoreRAF = null;

function interpAt(centers, vals, f) {
  if (f <= centers[0]) return vals[0];
  if (f >= centers[centers.length - 1]) return vals[vals.length - 1];
  for (let i = 1; i < centers.length; i += 1) {
    if (f <= centers[i]) {
      const t = (f - centers[i - 1]) / (centers[i] - centers[i - 1] || 1);
      return vals[i - 1] + t * (vals[i] - vals[i - 1]);
    }
  }
  return vals[vals.length - 1];
}

function gapBadge(label) {
  return label === "wrong"
    ? `<span class="badge badgeBad">❌ impossible ≤ possible</span>`
    : `<span class="badge badgeGood">✅ impossible &gt; possible</span>`;
}

// --- Magma colormap + per-patch overlay drawing for the violation window ---

const MAGMA = [
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
];

function magma(t) {
  const x = Math.max(0, Math.min(1, t)) * (MAGMA.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = MAGMA[i];
  const b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
  return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
}

// Draw the per-patch mask outline and/or surprise heatmap for ONE window's 16x16
// map onto a canvas sized over the video. The model's token grid covers only the
// center `inset` fraction of the encoded frame, so patches are mapped into that
// inset rectangle — the overlay aligns with the tokens, not the full frame.
function drawOverlay(canvas, video, hm, grid, mask, showMask, showHeat) {
  const W = canvas.width = video.clientWidth || 256;
  const H = canvas.height = video.clientHeight || 256;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  if (!grid) return;
  const inset = hm.overlay_inset || 0;
  const rows = hm.grid_h, cols = hm.grid_w;
  const x0 = inset * W, y0 = inset * H, gw = (1 - 2 * inset) * W, gh = (1 - 2 * inset) * H;
  const range = Math.max(1e-9, hm.vmax - hm.vmin);
  const pw = gw / cols, ph = gh / rows;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const px = x0 + c * pw, py = y0 + r * ph;
      if (showHeat) {
        const [R, G, B] = magma((grid[r][c] - hm.vmin) / range);
        ctx.fillStyle = `rgba(${R},${G},${B},0.55)`;
        ctx.fillRect(px, py, pw + 0.5, ph + 0.5);
      }
      if (showMask && mask && mask[r][c]) {
        ctx.strokeStyle = "rgba(80,255,220,0.95)";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(px + 0.75, py + 0.75, pw - 1.5, ph - 1.5);
      }
    }
  }
}

function overlayToggles(example) {
  if (!example.heatmap) return "";
  const n = example.heatmap.center.length;
  return `<div class="ovlToggles">
    <label><input type="checkbox" class="tgPdiff" checked> pixel-diff (chart)</label>
    <label><input type="checkbox" class="tgMask" checked> patches that differ</label>
    <label><input type="checkbox" class="tgHeat"> surprise heatmap</label>
    <span class="ovlHint">heatmap &amp; mask follow the playhead across the whole clip (${n} windows); the orange chart band marks where the two clips differ in pixels — <em>not</em> necessarily the violation</span>
  </div>`;
}

function videoFigure(side, cls, tag, src) {
  return `<figure><figcaption style="color:${side === "Possible" ? "#2364aa" : "#bd3c3c"}">${side}</figcaption>
    <div class="videoWrap"><span class="frameTag ${tag}">frame –</span>
      <video class="${cls}" muted playsinline preload="auto" src="${src}"></video>
      <canvas class="ovl ${cls}Ovl"></canvas></div></figure>`;
}

function curationPanel(example) {
  const item = inventoryItemFor(example.id);
  if (!item) return "";
  const status = item.visual_status || "unreviewed";
  // On the static public site the curation fields are read-only (no backend to
  // persist to) — disable the inputs and drop the Save button.
  const ro = STATIC_BASE ? " disabled" : "";
  const options = [
    ["unreviewed", "needs review"],
    ["reviewed_clean", "reviewed: clean"],
    ["visual_fail", "visual fail"],
    ["mask_issue", "mask issue"],
    ["motion_confound", "motion confound"],
    ["metadata_issue", "metadata issue"],
  ];
  return `<section class="curationPanel">
    <div class="curationHead">
      <h3>Human Review</h3>
      <span class="curationSaveState">${STATIC_BASE ? "read-only" : statusLabel(status)}</span>
    </div>
    <div class="curationGrid">
      <label>Status
        <select class="curationStatus"${ro}>
          ${options.map(([value, label]) => `<option value="${value}" ${value === status ? "selected" : ""}>${label}</option>`).join("")}
        </select>
      </label>
      <label>Violation type
        <input class="curationViolation" value="${item.violation_type || ""}" placeholder="unknown"${ro} />
      </label>
      <label>Object type
        <input class="curationObject" value="${item.object_type || ""}" placeholder="unknown"${ro} />
      </label>
    </div>
    <label class="curationNotes">Notes
      <textarea class="curationNoteText" rows="3" placeholder="What did you see?"${ro}>${item.visual_notes || ""}</textarea>
    </label>
    ${STATIC_BASE ? "" : `<button class="curationSave" type="button">Save review</button>`}
  </section>`;
}

function renderRescorePlayer(example) {
  els.exampleDetail.className = "";
  els.exampleDetail.innerHTML = `
    <div class="rescoreHead">
      <h2>${example.id} ${gapBadge(example.label)}</h2>
      <div class="rescoreControls">
        <button class="rescorePlay">▶ play</button>
        <input class="rescoreScrub" type="range" min="0" max="${example.n_frames - 1}" value="0" step="1" />
        <span class="rescoreReadout"></span>
      </div>
    </div>
    ${overlayToggles(example)}
    <div class="rescoreVideos">
      ${videoFigure("Possible", "vPossible", "pTag", assetUrl(example.video_possible))}
      ${videoFigure("Impossible", "vImpossible", "iTag", assetUrl(example.video_impossible))}
    </div>
    <svg class="rescoreChart" viewBox="0 0 600 240"></svg>
    ${metricGrid(example.metrics)}
    ${curationPanel(example)}`;
  initRescorePlayer(els.exampleDetail, example);
  initCurationPanel(els.exampleDetail, example);
}

async function saveInventoryItem(spec, updates) {
  const res = await fetch(`/api/inventory/item?run=${encodeURIComponent(state.runId)}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({spec, updates}),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  const idx = (state.inventory?.items || []).findIndex((item) => item.spec === spec);
  if (idx >= 0) state.inventory.items[idx] = data.item;
  return data.item;
}

function initCurationPanel(root, example) {
  const panel = root.querySelector(".curationPanel");
  if (!panel) return;
  const save = panel.querySelector(".curationSave");
  if (!save) return;  // static (read-only) build has no Save button
  const stateEl = panel.querySelector(".curationSaveState");
  save.addEventListener("click", async () => {
    save.disabled = true;
    stateEl.textContent = "saving...";
    try {
      const item = await saveInventoryItem(example.id, {
        visual_status: panel.querySelector(".curationStatus").value,
        visual_notes: panel.querySelector(".curationNoteText").value,
        violation_type: panel.querySelector(".curationViolation").value,
        object_type: panel.querySelector(".curationObject").value,
      });
      stateEl.textContent = `saved: ${statusLabel(item.visual_status)}`;
      renderInventory();
    } catch (error) {
      stateEl.textContent = "save failed";
      console.error(error);
    } finally {
      save.disabled = false;
    }
  });
}

function initRescorePlayer(root, example) {
  if (rescoreRAF) cancelAnimationFrame(rescoreRAF);
  const c = example.dense_curve;
  const nF = example.n_frames;
  const fps = example.fps || 12;
  const pos = root.querySelector(".vPossible");
  const imp = root.querySelector(".vImpossible");
  const playBtn = root.querySelector(".rescorePlay");
  const scrub = root.querySelector(".rescoreScrub");
  const readout = root.querySelector(".rescoreReadout");
  const pTag = root.querySelector(".pTag");
  const iTag = root.querySelector(".iTag");
  const svg = root.querySelector(".rescoreChart");
  const frameOf = (v) => Math.min(nF - 1, Math.max(0, Math.round(v.currentTime * fps)));

  const W = 600, H = 240, pad = 40;
  const ys = [...c.possible, ...c.impossible];
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  const padY = (maxY - minY) * 0.15 || 0.005;
  minY -= padY; maxY += padY;
  const sx = (f) => pad + (f / Math.max(1, nF - 1)) * (W - pad * 2);
  const sy = (v) => H - pad - ((v - minY) / Math.max(1e-9, maxY - minY)) * (H - pad * 2);
  // The surprise curve only spans window centers (the first is ~frame 11.5, since
  // a window needs context frames before it), but the playhead spans the whole
  // clip (frame 0..nF-1). Hold the first/last value flat out to the chart edges so
  // the line covers the full width — this matches interpAt's clamping exactly, so
  // the playhead dot rides the line at every frame instead of floating in the gap.
  const poly = (vals, color) => {
    const pts = [
      `${sx(0)},${sy(vals[0])}`,
      ...c.center.map((f, i) => `${sx(f)},${sy(vals[i])}`),
      `${sx(nF - 1)},${sy(vals[vals.length - 1])}`,
    ].join(" ");
    return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}" />`;
  };
  // Honest "where the two clips differ in pixels" band, normalized to its own
  // max and drawn faintly along the bottom — explicitly NOT a violation marker.
  const pd = example.pixel_diff;
  let pdArea = "";
  if (pd && pd.value.length) {
    const pmax = Math.max(1e-9, ...pd.value);
    const baseY = H - pad, topBand = baseY - (H - 2 * pad) * 0.3;
    const pts = pd.center.map((f, i) => `${sx(f)},${baseY - (pd.value[i] / pmax) * (baseY - topBand)}`).join(" ");
    pdArea = `<polygon class="pdArea" points="${sx(pd.center[0])},${baseY} ${pts} ${sx(pd.center[pd.center.length - 1])},${baseY}" fill="#9a6a13" opacity="0.16" />`;
  }
  svg.innerHTML = `
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#d9dee6" />
    ${pdArea}
    <text x="${W - pad}" y="${H - 12}" text-anchor="end" class="axisLabel">clip time →</text>
    <text x="${pad}" y="${pad - 12}" class="axisLabel" style="fill:#2364aa">possible</text>
    <text x="${pad + 70}" y="${pad - 12}" class="axisLabel" style="fill:#bd3c3c">impossible</text>
    <text x="${W - pad}" y="${pad - 12}" text-anchor="end" class="axisLabel" style="fill:#9a6a13">pixel diff</text>
    ${poly(c.possible, "#2364aa")}${poly(c.impossible, "#bd3c3c")}
    <line class="playLine" x1="${sx(0)}" y1="${pad - 6}" x2="${sx(0)}" y2="${H - pad}" stroke="#444" stroke-width="1" />
    <circle class="playDot dotP" r="6" cx="${sx(0)}" cy="${sy(c.possible[0])}" fill="#2364aa" />
    <circle class="playDot dotI" r="6" cx="${sx(0)}" cy="${sy(c.impossible[0])}" fill="#bd3c3c" />`;
  const playLine = svg.querySelector(".playLine");
  const dotP = svg.querySelector(".dotP");
  const dotI = svg.querySelector(".dotI");
  let drawOv = null;  // assigned below if this example carries overlay data

  function place(frame) {
    const x = sx(frame);
    const vp = interpAt(c.center, c.possible, frame);
    const vi = interpAt(c.center, c.impossible, frame);
    playLine.setAttribute("x1", x); playLine.setAttribute("x2", x);
    dotP.setAttribute("cx", x); dotP.setAttribute("cy", sy(vp));
    dotI.setAttribute("cx", x); dotI.setAttribute("cy", sy(vi));
    readout.textContent = `frame ${Math.round(frame)}/${nF - 1} · possible ${vp.toFixed(4)} · impossible ${vi.toFixed(4)}`;
    if (drawOv) drawOv(frame);
  }

  function loop() {
    if (!root.isConnected) return;
    const frame = pos.currentTime * fps;
    if (Math.abs(imp.currentTime - pos.currentTime) > 0.08) imp.currentTime = pos.currentTime;
    scrub.value = String(Math.min(nF - 1, Math.round(frame)));
    place(frame);
    // Each tag reads its OWN video's frame, so any drift between the two
    // clips (or the chart) shows up rather than being masked.
    pTag.textContent = `frame ${frameOf(pos)}`;
    iTag.textContent = `frame ${frameOf(imp)}`;
    rescoreRAF = requestAnimationFrame(loop);
  }
  function play() {
    pos.play(); imp.play();
    playBtn.textContent = "⏸ pause";
    rescoreRAF = requestAnimationFrame(loop);
  }
  function stop() {
    pos.pause(); imp.pause();
    if (rescoreRAF) cancelAnimationFrame(rescoreRAF);
    rescoreRAF = null;
    playBtn.textContent = "▶ play";
  }
  playBtn.addEventListener("click", () => (rescoreRAF ? stop() : play()));
  scrub.addEventListener("input", () => {
    const f = Number(scrub.value);
    const t = f / fps;
    pos.currentTime = t; imp.currentTime = t;
    place(f);
    pTag.textContent = `frame ${f}`;
    iTag.textContent = `frame ${f}`;
  });
  pos.addEventListener("ended", () => {
    stop(); pos.currentTime = 0; imp.currentTime = 0; place(0); scrub.value = "0";
    pTag.textContent = "frame 0"; iTag.textContent = "frame 0";
  });

  if (example.heatmap) {
    const hm = example.heatmap;
    const pOvl = root.querySelector(".vPossibleOvl");
    const iOvl = root.querySelector(".vImpossibleOvl");
    const tgPdiff = root.querySelector(".tgPdiff");
    const tgMask = root.querySelector(".tgMask");
    const tgHeat = root.querySelector(".tgHeat");
    const pdEl = svg.querySelector(".pdArea");
    // Nearest heatmap window to the current frame — overlay spans the whole clip,
    // no pre-selected window.
    const nearestWindow = (f) => {
      let best = 0, bestD = Infinity;
      for (let i = 0; i < hm.center.length; i += 1) {
        const d = Math.abs(hm.center[i] - f);
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    };
    drawOv = (frame) => {
      if (pdEl) pdEl.style.display = tgPdiff.checked ? "" : "none";
      const j = nearestWindow(frame);
      const showMask = tgMask.checked, showHeat = tgHeat.checked;
      drawOverlay(pOvl, pos, hm, hm.possible[j], hm.mask[j], showMask, showHeat);
      drawOverlay(iOvl, imp, hm, hm.impossible[j], hm.mask[j], showMask, showHeat);
    };
    [tgPdiff, tgMask, tgHeat].forEach((t) => t.addEventListener("change", () => drawOv(Number(scrub.value))));
    // Video intrinsic size isn't known until metadata loads; redraw then.
    pos.addEventListener("loadeddata", () => drawOv(Number(scrub.value)));
    imp.addEventListener("loadeddata", () => drawOv(Number(scrub.value)));
  }

  place(0);
  pTag.textContent = "frame 0";
  iTag.textContent = "frame 0";
}

function renderExampleDetail() {
  const example = state.examples.find((item) => item.id === state.selectedExampleId);
  if (!example) {
    els.exampleDetail.className = "empty";
    els.exampleDetail.textContent = state.examples.length ? "Select a pair." : "No pairs in this run.";
    return;
  }
  if (example.video_possible) {
    renderRescorePlayer(example);
    return;
  }
  // The only supported example shape is the dense video re-score. Older runs
  // (frame-strip pair format) need the rescore adapter re-run to view here.
  els.exampleDetail.className = "empty";
  els.exampleDetail.innerHTML =
    `<div>${example.id}: no video for this pair.<br />` +
    `Re-run <code>viewer.adapters.intphys_rescore</code> on this run to produce the side-by-side player.</div>`;
}

// --- Population tab: the headline VoE result, made browsable ---

function populationRows() {
  return state.examples
    .filter((e) => typeof e.metrics?.surprise_gap === "number")
    .map((e) => ({
      id: e.id,
      block: blockOf(e),
      gap: e.metrics.surprise_gap,
      correct: e.label !== "wrong" && e.metrics.surprise_gap > 0,
    }));
}

function gapHistogram(rows) {
  const gaps = rows.map((r) => r.gap);
  const absMax = Math.max(1e-6, ...gaps.map((g) => Math.abs(g)));
  const nBins = 21; // odd so a bin is centered on zero
  const counts = new Array(nBins).fill(0);
  const edge = absMax * 1.0001;
  for (const g of gaps) {
    let bin = Math.floor(((g + edge) / (2 * edge)) * nBins);
    bin = Math.min(nBins - 1, Math.max(0, bin));
    counts[bin] += 1;
  }
  const W = 600, H = 180, pad = 28;
  const maxC = Math.max(1, ...counts);
  const bw = (W - pad * 2) / nBins;
  const zeroX = pad + (W - pad * 2) / 2;
  const bars = counts
    .map((c, i) => {
      const h = (c / maxC) * (H - pad * 2);
      const x = pad + i * bw;
      const center = -edge + (i + 0.5) * (2 * edge) / nBins;
      const fill = center > 0 ? "#bd3c3c" : center < 0 ? "#2364aa" : "#888";
      return `<rect x="${x + 1}" y="${H - pad - h}" width="${Math.max(1, bw - 2)}" height="${h}" fill="${fill}" opacity="0.85"><title>gap≈${center.toFixed(4)} · ${c} pair(s)</title></rect>`;
    })
    .join("");
  return `<svg class="popHist" viewBox="0 0 ${W} ${H}">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#d9dee6" />
    <line x1="${zeroX}" y1="${pad - 6}" x2="${zeroX}" y2="${H - pad}" stroke="#444" stroke-dasharray="4 3" />
    <text x="${zeroX}" y="${pad - 10}" text-anchor="middle" class="axisLabel">gap = 0</text>
    <text x="${W - pad}" y="${H - 8}" text-anchor="end" class="axisLabel" style="fill:#bd3c3c">impossible more surprising →</text>
    <text x="${pad}" y="${H - 8}" class="axisLabel" style="fill:#2364aa">← possible more surprising</text>
    ${bars}</svg>`;
}

function blockTable(rows) {
  const byBlock = {};
  for (const r of rows) {
    const b = (byBlock[r.block] ||= { n: 0, correct: 0 });
    b.n += 1;
    if (r.correct) b.correct += 1;
  }
  const blocks = Object.keys(byBlock).sort();
  if (blocks.length <= 1) return "";
  return `<table class="popBlocks">
    <thead><tr><th>block</th><th>pairs</th><th>correct</th><th>accuracy</th></tr></thead>
    <tbody>${blocks
      .map((b) => {
        const { n, correct } = byBlock[b];
        return `<tr><td>${b}</td><td>${n}</td><td>${correct}</td><td>${(correct / n).toFixed(3)}</td></tr>`;
      })
      .join("")}</tbody></table>`;
}

function sortPopRows(rows) {
  const s = state.popSort;
  const copy = [...rows];
  if (s === "absgap") copy.sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
  else if (s === "gap") copy.sort((a, b) => b.gap - a.gap);
  else if (s === "id") copy.sort((a, b) => a.id.localeCompare(b.id));
  return copy;
}

// The mirror plot: each IntPhys scene contributes two matched pairs; plotting
// pair-A gap vs pair-B gap shows them piling on the dashed y = −x line where the
// two gaps cancel — the visual proof that the metric reads appearance, not
// possibility. Off-line points (orange) are the rare scenes that didn't cancel.
function mirrorPlot(analysis) {
  const m = analysis.mirror || [];
  if (!m.length) return "";
  const W = 440, H = 440, pad = 46;
  const lim = Math.max(1e-6, ...m.flat().map(Math.abs)) * 1.05;
  const sx = (v) => pad + ((v + lim) / (2 * lim)) * (W - 2 * pad);
  const sy = (v) => H - pad - ((v + lim) / (2 * lim)) * (H - 2 * pad);
  const pts = m
    .map(([a, b]) => {
      const anom = (a > 0) === (b > 0); // same sign => did not cancel
      return `<circle cx="${sx(a)}" cy="${sy(b)}" r="4" fill="${anom ? "#9a6a13" : "#bd3c3c"}" opacity="0.7"><title>gap_a ${a.toFixed(4)}, gap_b ${b.toFixed(4)}${anom ? " — did NOT cancel" : ""}</title></circle>`;
    })
    .join("");
  return `<svg class="mirrorPlot" viewBox="0 0 ${W} ${H}">
    <line x1="${sx(-lim)}" y1="${sy(lim)}" x2="${sx(lim)}" y2="${sy(-lim)}" stroke="#444" stroke-dasharray="5 4" />
    <line x1="${pad}" y1="${sy(0)}" x2="${W - pad}" y2="${sy(0)}" stroke="#e1e5ea" />
    <line x1="${sx(0)}" y1="${pad}" x2="${sx(0)}" y2="${H - pad}" stroke="#e1e5ea" />
    <text x="${sx(lim)}" y="${sy(-lim) + 16}" text-anchor="end" class="axisLabel">y = −x (gaps cancel)</text>
    <text x="${W - pad}" y="${sy(0) - 6}" text-anchor="end" class="axisLabel">pair A gap →</text>
    <text x="${sx(0) + 6}" y="${pad - 2}" class="axisLabel">pair B gap ↑</text>
    ${pts}</svg>`;
}

function analysisSummary(a) {
  const beats = a.accuracy_localized > a.accuracy_motion;
  return `<div class="popHeadline">
      <div class="popAcc">
        <span class="popAccBig">${a.accuracy_localized.toFixed(3)}</span>
        <span class="popAccSub">${a.n_pairs} pairs · violation-localized</span>
      </div>
      <div class="popAccNote">
        Model-free <strong>motion baseline ${a.accuracy_motion.toFixed(3)}</strong> — the localized
        surprise ${beats ? "beats" : "does <strong>not</strong> beat"} it. Within-scene anti-symmetry
        <strong>${(a.anti_symmetry_pct * 100).toFixed(0)}%</strong>: in most scenes the two pair-gaps
        are mirror images that cancel, so the metric reads appearance, not physics. Each dot below is
        one scene's two gaps; they pile on the dashed <em>y = −x</em> line where they cancel. The few
        orange points are scenes that didn't.
      </div>
    </div>
    <div class="mirrorWrap">${mirrorPlot(a)}</div>`;
}

function renderPopulation() {
  const rows = populationRows();
  const analysis = state.manifest?.analysis;
  els.populationEmpty.style.display = rows.length ? "none" : "grid";
  els.populationBody.style.display = rows.length ? "block" : "none";
  if (!rows.length) {
    els.populationBody.innerHTML = "";
    return;
  }
  const run = state.manifest?.run || {};
  const provenance = [
    run.commit ? `commit ${run.commit}` : "",
    run.command ? `<code>${run.command}</code>` : "",
  ].filter(Boolean).join(" · ");

  const list = sortPopRows(rows)
    .map(
      (r) => `<button class="popRow" data-example="${r.id}">
        <span class="popId">${r.id}</span>
        <span class="popBlock">${r.block}</span>
        <span class="popGap ${r.gap > 0 ? "pos" : "neg"}">${r.gap >= 0 ? "+" : ""}${r.gap.toFixed(4)}</span>
        <span class="popMark">${r.correct ? "✓" : "✗"}</span>
      </button>`
    )
    .join("");
  const sortControl = `<div class="popListHead">
      <span>${analysis ? "curated examples — click to inspect with overlays" : rows.length + " pairs · click a row to open its player"}</span>
      <label>sort
        <select id="popSortSel">
          <option value="absgap">|gap| (most decisive)</option>
          <option value="gap">gap (impossible − possible)</option>
          <option value="id">id</option>
        </select>
      </label>
    </div>`;

  let head;
  if (analysis) {
    head = analysisSummary(analysis);
  } else {
    const n = rows.length;
    const nCorrect = rows.filter((r) => r.correct).length;
    head = `<div class="popHeadline">
        <div class="popAcc">
          <span class="popAccBig">${(nCorrect / n).toFixed(3)}</span>
          <span class="popAccSub">${nCorrect}/${n} pairs · impossible &gt; possible</span>
        </div>
        <div class="popAccNote">
          Chance is <strong>0.500</strong>. VoE accuracy near chance means ViT-L surprise barely
          separates physically impossible from possible — the headline negative result, per SOUL.md.
          Gaps are tiny: the distribution below is centered on zero.
        </div>
      </div>
      ${gapHistogram(rows)}
      ${blockTable(rows)}`;
  }

  els.populationBody.innerHTML = head + sortControl
    + `<div class="popList">${list}</div>`
    + (provenance ? `<p class="popProvenance">${provenance}</p>` : "");

  const sel = els.populationBody.querySelector("#popSortSel");
  if (sel) {
    sel.value = state.popSort;
    sel.addEventListener("change", () => {
      state.popSort = sel.value;
      renderPopulation();
    });
  }
}

// --- Inventory tab: curation queue + quantitative diagnostics ---

function inventoryItems() {
  const filter = (els.inventoryFilter?.value || "").trim().toLowerCase();
  const view = els.inventoryView?.value || "all";
  return [...(state.inventory?.items || [])]
    .filter((item) => {
      if (view === "motion" && !item.motion_agrees_with_localized) return false;
      if (view === "unreviewed" && item.visual_status !== "unreviewed") return false;
      if (view === "metadata" && item.metadata_quality === "complete") return false;
      if (!filter) return true;
      const haystack = [
        item.spec,
        item.family,
        item.scene,
        item.motion_direction,
        item.scene_pair_sign,
        item.metadata_quality,
        item.visual_status,
        item.violation_type,
        item.object_type,
      ].join(" ").toLowerCase();
      return haystack.includes(filter);
    })
    .sort((a, b) => {
      if (a.motion_agrees_with_localized !== b.motion_agrees_with_localized) {
        return a.motion_agrees_with_localized ? -1 : 1;
      }
      return Math.abs(b.localized_gap || 0) - Math.abs(a.localized_gap || 0);
    });
}

function renderInventory() {
  const summary = state.inventory?.summary || {};
  const items = inventoryItems();
  const hasInventory = Boolean(state.inventory?.items?.length);
  els.inventoryEmpty.style.display = hasInventory ? "none" : "grid";
  els.inventoryBody.style.display = hasInventory ? "block" : "none";
  if (!hasInventory) {
    els.inventoryBody.innerHTML = "";
    return;
  }

  const familyText = Object.entries(summary.families || {})
    .map(([key, value]) => `${key}: ${value}`)
    .join(" · ");
  const motionAligned = summary.motion_agrees_with_localized || 0;
  const maskAvailable = summary.mask_available || 0;
  const rows = items.map((item) => {
    const metadataBad = item.metadata_quality !== "complete";
    const reviewStatus = statusLabel(item.visual_status);
    return `<button class="invRow" data-example="${item.spec}">
        <span class="invSpec">${item.spec}</span>
        <span>${item.family}</span>
        <span>${item.motion_agrees_with_localized ? "motion-aligned" : "motion-opposes"}</span>
        <span>${formatValue(item.localized_gap)}</span>
        <span>${formatValue(item.motion_diff)}</span>
        <span>${item.mask_available ? `${item.mask_active_peak} peak` : "none"}</span>
        <span class="${metadataBad ? "invWarn" : ""}">${item.metadata_quality}</span>
        <span>${reviewStatus}</span>
      </button>`;
  }).join("");

  els.inventoryBody.innerHTML = `
    <div class="invSummary">
      <span class="statChip"><strong>${summary.n_items || 0}</strong> inventory cases</span>
      <span class="statChip">${familyText || "families unknown"}</span>
      <span class="statChip"><strong>${maskAvailable}</strong> with masks</span>
      <span class="statChip"><strong>${motionAligned}</strong> motion-aligned candidates</span>
    </div>
    <div class="invHead">
      <span>${items.length} shown · human review fields start empty until curated</span>
      <span><code>inventory.json</code> + <code>inventory.csv</code></span>
    </div>
    <div class="invTable">
      <div class="invHeader">
        <span>case</span><span>family</span><span>motion</span><span>localized gap</span>
        <span>motion diff</span><span>mask</span><span>metadata</span><span>human review</span>
      </div>
      ${rows || `<div class="empty">No rows match this filter.</div>`}
    </div>`;
}

function renderHome() {
  const run = state.manifest?.run || {};
  const a = state.manifest?.analysis;
  if (a) {
    const reviewed = state.examples?.length || 0;
    els.homeStats.innerHTML = `
      <span class="statChip">Current run: <strong>${run.id || "—"}</strong></span>
      <span class="statChip">${reviewed} reviewed pair${reviewed === 1 ? "" : "s"}</span>
      <span class="statChip">${a.n_pairs} analysis pairs</span>
      <span class="statChip">violation-localized <strong>${a.accuracy_localized.toFixed(3)}</strong> vs motion <strong>${a.accuracy_motion.toFixed(3)}</strong></span>
      <span class="statChip">within-scene anti-symmetry <strong>${(a.anti_symmetry_pct * 100).toFixed(0)}%</strong></span>`;
    return;
  }
  const rows = populationRows();
  if (!rows.length) {
    els.homeStats.innerHTML = `<span class="statChip">Current run: <strong>${run.id || "—"}</strong> · no scored pairs</span>`;
    return;
  }
  const n = rows.length;
  const nCorrect = rows.filter((r) => r.correct).length;
  const acc = (nCorrect / n).toFixed(3);
  els.homeStats.innerHTML = `
    <span class="statChip">Current run: <strong>${run.id || "—"}</strong></span>
    <span class="statChip">${n} pairs</span>
    <span class="statChip">VoE accuracy <strong>${acc}</strong> <em>(chance 0.500)</em></span>`;
}

function renderAll() {
  state.examples = state.manifest?.examples || [];
  renderRunMeta();
  renderSortControls();
  renderExampleList();
  renderExampleDetail();
  renderHome();
  renderPopulation();
  renderInventory();
}

async function loadRuns() {
  const res = await fetch(STATIC_BASE ? `${STATIC_BASE}/runs.json` : "/api/runs");
  const data = await res.json();
  state.runs = data.runs || [];
  els.runSelect.innerHTML = state.runs.map((run) => `<option value="${run.id}">${run.id}</option>`).join("");
  state.runId = state.runs[0]?.id;
}

async function loadManifest(runId) {
  let data, invData;
  if (STATIC_BASE) {
    // Static build: the API responses are pre-baked as plain files. The manifest
    // file is the bare manifest; the API form wraps it as {run_id, manifest}.
    const mres = await fetch(`${STATIC_BASE}/${encodeURIComponent(runId)}/manifest.json`);
    if (!mres.ok) throw new Error(await mres.text());
    data = { run_id: runId, manifest: await mres.json() };
    const ires = await fetch(`${STATIC_BASE}/${encodeURIComponent(runId)}/inventory.json`);
    invData = { inventory: ires.ok ? await ires.json() : null };
  } else {
    const res = await fetch(`/api/manifest?run=${encodeURIComponent(runId)}`);
    if (!res.ok) throw new Error(await res.text());
    data = await res.json();
    const invRes = await fetch(`/api/inventory?run=${encodeURIComponent(runId)}`);
    invData = invRes.ok ? await invRes.json() : { inventory: null };
  }
  state.runId = data.run_id;
  state.manifest = data.manifest;
  state.inventory = invData.inventory;
  state.selectedExampleId = null;
  renderAll();
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === name));
}

document.querySelector(".tabs").addEventListener("click", (event) => {
  const button = event.target.closest(".tab");
  if (!button) return;
  activateTab(button.dataset.tab);
});

els.runSelect.addEventListener("change", () => loadManifest(els.runSelect.value));
els.exampleFilter.addEventListener("input", renderExampleList);
els.metricSort.addEventListener("change", renderExampleList);
els.inventoryFilter.addEventListener("input", renderInventory);
els.inventoryView.addEventListener("change", renderInventory);
els.exampleList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  state.selectedExampleId = button.dataset.example;
  renderExampleList();
  renderExampleDetail();
});
els.populationBody.addEventListener("click", (event) => {
  const row = event.target.closest("[data-example]");
  if (!row) return;
  state.selectedExampleId = row.dataset.example;
  activateTab("examples");
  renderExampleList();
  renderExampleDetail();
});
els.inventoryBody.addEventListener("click", (event) => {
  const row = event.target.closest("[data-example]");
  if (!row) return;
  state.selectedExampleId = row.dataset.example;
  activateTab("examples");
  renderExampleList();
  renderExampleDetail();
});

loadRuns()
  .then(() => loadManifest(state.runId))
  .catch((error) => {
    els.runMeta.textContent = error.message;
  });
