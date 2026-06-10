const state = {
  runs: [],
  runId: null,
  manifest: null,
  inventory: null,
  examples: [],
  selectedExampleId: null,
  exploreView: "latent",
  popView: "probes",
};

const els = {
  runMeta: document.querySelector("#runMeta"),
  runSelect: document.querySelector("#runSelect"),
  exampleFilter: document.querySelector("#exampleFilter"),
  metricSort: document.querySelector("#metricSort"),
  exampleList: document.querySelector("#exampleList"),
  exampleDetail: document.querySelector("#exampleDetail"),
  populationDetail: document.querySelector("#populationDetail"),
  homeStats: document.querySelector("#homeStats"),
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

// --- Examples tab (latent_surface runs): the same lockstep player, but the curve
// is replaced by a latent SURFACE — a shared-PCA trajectory plus rails for
// effective rank, latent-velocity-vs-pixel-flow, and possible/impossible
// divergence drawn against its own shuffled-pair null band. ---

let latentRAF = null;

// hold-flat-to-edges polyline points, matching interpAt's clamping so the
// playhead dot rides the line across the whole clip (same trick as the surprise
// chart). `sx`/`sy` are screen scales; `centers` are window-center frames.
function holdPoly(centers, vals, nF, sx, sy) {
  return [
    `${sx(0)},${sy(vals[0])}`,
    ...centers.map((f, i) => `${sx(f)},${sy(vals[i])}`),
    `${sx(nF - 1)},${sy(vals[vals.length - 1])}`,
  ].join(" ");
}

// One stacked rail: a small line chart sharing the clip-time x-axis and playhead.
// `series` = [{vals, color, dash}]; optional `band` = {index, lo, hi} shaded behind.
// `normalize` rescales each series to its own [0,1] so signals in *different units*
// (latent velocity vs pixel flow) are shape-comparable instead of one squashing
// the other; raw same-unit rails (rank, divergence) leave it off.
function railSVG(cls, title, centers, nF, series, band, normalize) {
  const W = 600, H = 110, padX = 40, padT = 12, padB = 20;
  if (normalize) {
    series = series.map((s) => {
      const mx = Math.max(1e-9, ...s.vals);
      return {...s, vals: s.vals.map((v) => v / mx)};
    });
  }
  const all = series.flatMap((s) => s.vals);
  if (band) all.push(...band.lo, ...band.hi);
  let minY = Math.min(...all), maxY = Math.max(...all);
  const padY = (maxY - minY) * 0.12 || 0.01;
  minY -= padY; maxY += padY;
  const sx = (f) => padX + (f / Math.max(1, nF - 1)) * (W - padX * 2);
  const sy = (v) => H - padB - ((v - minY) / Math.max(1e-9, maxY - minY)) * (H - padT - padB);
  let bandPoly = "";
  if (band && band.index.length) {
    const cx = (k) => sx(centers[Math.min(centers.length - 1, k)]);
    const top = band.index.map((k, i) => `${cx(k)},${sy(band.hi[i])}`);
    const bot = band.index.map((k, i) => `${cx(k)},${sy(band.lo[i])}`).reverse();
    bandPoly = `<polygon class="nullBand" points="${[...top, ...bot].join(" ")}" />`;
  }
  const lines = series
    .map((s) => `<polyline class="railLine" fill="none" stroke="${s.color}" stroke-width="2"`
      + `${s.dash ? ` stroke-dasharray="${s.dash}"` : ""} points="${holdPoly(centers, s.vals, nF, sx, sy)}" />`)
    .join("");
  return `<figure class="chartCard">
    <figcaption>${title}</figcaption>
    <svg class="railChart ${cls}" viewBox="0 0 ${W} ${H}" data-miny="${minY}" data-maxy="${maxY}">
      ${bandPoly}${lines}
      <line class="railPlay" x1="${sx(0)}" y1="${padT - 6}" x2="${sx(0)}" y2="${H - padB}" />
    </svg></figure>`;
}

// --- delta-direction views (#1 cosine matrix, #2 delta-PCA, #3 pairwise, #5 loadings) ---

const BLOCK_COLORS = {O1: "#2364aa", O2: "#bd3c3c", O3: "#7a3ea8"};
const blockColor = (id) => BLOCK_COLORS[String(id).split(":")[0]] || "#888";

// Diverging blue(−1) → white(0) → red(+1) for cosine values.
function divColor(v) {
  const t = Math.min(1, Math.abs(Math.max(-1, Math.min(1, v))));
  const c = v < 0 ? [35, 100, 170] : [189, 60, 60];
  const m = (x) => Math.round(255 + (x - 255) * t);
  return `rgb(${m(c[0])},${m(c[1])},${m(c[2])})`;
}

// #2 — each per-pair Δ direction as a point in the PCA-of-deltas; colored by block,
// selected pair ringed. Clicking a dot opens that pair.
function deltaScatterSVG(da, selId) {
  const VB = 300, pad = 22, padTop = 38;
  const xy = da.delta_pca.xy;
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (x) => pad + ((x - xmin) / ((xmax - xmin) || 1)) * (VB - 2 * pad);
  const py = (y) => VB - pad - ((y - ymin) / ((ymax - ymin) || 1)) * (VB - pad - padTop);
  const dots = da.specs.map((s, i) => {
    const sel = s === selId;
    return `<circle class="dScatterDot" data-spec="${s}" cx="${px(xy[i][0])}" cy="${py(xy[i][1])}" r="${sel ? 6 : 4}" fill="${blockColor(s)}" stroke="${sel ? "#111" : "#fff"}" stroke-width="${sel ? 2 : 1}"><title>${s}</title></circle>`;
  }).join("");
  return `<figure class="chartCard">
    <figcaption>Each dot is one clip's impossible-event “nudge” — close dots were nudged the same way (color = scene type)</figcaption>
    <svg class="deltaScatter" viewBox="0 0 ${VB} ${VB}">${dots}</svg></figure>`;
}

// #1/#3 — full-dim cosine between every pair's Δ direction, as a heatmap. The
// selected pair's row/col is outlined; an overlay drives the hover readout (#3).
function cosMatrixSVG(da, selId) {
  const N = da.specs.length;
  const G = Math.min(360, Math.max(120, N * 7));
  const cell = G / N;
  let rects = "";
  for (let i = 0; i < N; i += 1) {
    for (let j = 0; j < N; j += 1) {
      rects += `<rect x="${j * cell}" y="${i * cell}" width="${cell + 0.6}" height="${cell + 0.6}" fill="${divColor(da.cosine_matrix[i][j])}" />`;
    }
  }
  const si = da.specs.indexOf(selId);
  const hl = si >= 0
    ? `<rect x="0" y="${si * cell}" width="${G}" height="${cell}" fill="none" stroke="#111" stroke-width="1.3"/>
       <rect x="${si * cell}" y="0" width="${cell}" height="${G}" fill="none" stroke="#111" stroke-width="1.3"/>`
    : "";
  return `<svg class="cosMatrix" viewBox="0 0 ${G} ${G}" data-n="${N}" data-grid="${G}">
    ${rects}${hl}
    <rect class="cosOverlay" x="0" y="0" width="${G}" height="${G}" fill="transparent" style="cursor:crosshair"/></svg>`;
}

// #3 — the −1..+1 cosine bar with the random-direction null band shaded around 0;
// a marker is moved as you hover the matrix.
function cosBarSVG(null95) {
  const W = 300, H = 34, pad = 10, midY = 20;
  const sx = (v) => pad + ((v + 1) / 2) * (W - 2 * pad);
  const band = `<rect x="${sx(-null95)}" y="${midY - 9}" width="${sx(null95) - sx(-null95)}" height="18" fill="#8893a3" opacity="0.25"/>`;
  return `<svg class="cosBar" viewBox="0 0 ${W} ${H}" data-w="${W}" data-pad="${pad}">
    <line x1="${pad}" y1="${midY}" x2="${W - pad}" y2="${midY}" stroke="#d9dee6"/>
    ${band}
    <text x="${pad}" y="${H - 2}" class="axisLabel" style="fill:#2364aa">−1 opposed</text>
    <text x="${W - pad}" y="${H - 2}" text-anchor="end" class="axisLabel" style="fill:#bd3c3c">aligned +1</text>
    <line class="cosMark" x1="${sx(0)}" y1="${midY - 11}" x2="${sx(0)}" y2="${midY + 11}" stroke="#111" stroke-width="2"/></svg>`;
}

// #5 — the pair's loading on each shared Δ-axis, as signed bars.
function loadingsSVG(L) {
  const ld = L.delta_loadings;
  if (!ld || !ld.length) return "";
  const W = 560, row = 22, H = row * ld.length + 8, padL = 56, padR = 12;
  const cx = (padL + (W - padR)) / 2, half = (W - padR - padL) / 2;
  const max = Math.max(0.3, ...ld.map(Math.abs));
  const bars = ld.map((v, k) => {
    const y = 4 + k * row, w = (Math.abs(v) / max) * half, x = v < 0 ? cx - w : cx;
    return `<text x="6" y="${y + 13}" class="svgAxis">axis ${k + 1}</text>
      <line x1="${cx}" y1="${y}" x2="${cx}" y2="${y + row - 6}" stroke="#ccc"/>
      <rect x="${x}" y="${y}" width="${w}" height="${row - 8}" fill="${v < 0 ? "#2364aa" : "#bd3c3c"}" opacity="0.82"><title>${v}</title></rect>`;
  }).join("");
  return `<figure class="chartCard">
    <figcaption>How this clip's nudge lines up with the main shared directions (axis 1 = biggest shared pattern)</figcaption>
    <svg class="loadingsChart" viewBox="0 0 ${W} ${H}">${bars}</svg></figure>`;
}

function renderLatentPlayer(example) {
  const L = example.latent;
  const nullBand = state.manifest?.latent_space?.null_divergence || null;
  const da = state.manifest?.latent_space?.delta_analysis || null;
  els.exampleDetail.className = "";
  const above = example.label === "above null";
  els.exampleDetail.innerHTML = `
    <div class="rescoreHead">
      <h2>${example.id} <span class="badge ${above ? "badgeGood" : "badgeBad"}">${example.label || "latent"}</span>
        <span class="viewName">· Latent surface</span></h2>
      <div class="rescoreControls">
        <button class="rescorePlay">▶ play</button>
        <input class="rescoreScrub" type="range" min="0" max="${example.n_frames - 1}" value="0" step="1" />
        <span class="rescoreReadout"></span>
      </div>
    </div>
    ${viewSwitcher("latent", example)}
    <p class="latentHint"><strong>In plain terms:</strong> the model turns each moment of video into
      a long list of numbers — its internal “read” of the scene. The map below squashes that to 2D so
      you can watch the <span style="color:#2364aa">normal</span> and
      <span style="color:#bd3c3c">impossible</span> clips as two moving dots. <strong>If the model
      notices the impossible event, the red dot should veer away from the blue one.</strong> The charts
      underneath track, moment by moment, different aspects of that read (each chart says what).</p>
    ${L.divergence_map ? `<div class="ovlToggles"><label><input type="checkbox" class="tgDiv">
      show where they differ on the frame — <span class="ovlHint">highlights the patches where the
      model's read of the normal vs impossible clip differs most, following the playhead</span></label></div>` : ""}
    <div class="latentTop">
      <div class="rescoreVideos latentVideos">
        ${videoFigure("Possible", "vPossible", "pTag", assetUrl(example.video_possible))}
        ${videoFigure("Impossible", "vImpossible", "iTag", assetUrl(example.video_impossible))}
      </div>
      <figure class="chartCard pcaCard">
        <figcaption>The model's “read”, squashed to 2D — each dot is one moment. If it notices the impossible event, red veers off from blue.</figcaption>
        <svg class="pcaPanel" viewBox="0 0 260 260"></svg>
      </figure>
    </div>
    <div class="rails">
      ${railSVG("railRank", "How many distinct features it's using (a dip = it simplified its read)", L.center, example.n_frames,
        [{vals: L.eff_rank.possible, color: "#2364aa"}, {vals: L.eff_rank.impossible, color: "#bd3c3c"}])}
      ${railSVG("railVel", "Speed the model's read is changing (solid) vs motion in the video (dashed) — solid jumping without dashed = it reshuffled even though little moved", L.center, example.n_frames,
        [{vals: L.latent_vel.possible, color: "#2364aa"}, {vals: L.latent_vel.impossible, color: "#bd3c3c"},
         {vals: L.flow.possible, color: "#2364aa", dash: "4 3"}, {vals: L.flow.impossible, color: "#bd3c3c", dash: "4 3"}], null, true)}
      ${railSVG("railDiv", "Gap between the normal & impossible clip (purple) vs the same-scene 'nothing unusual' baseline (grey) — purple above grey = treated as genuinely different", L.center, example.n_frames,
        [{vals: L.divergence, color: "#7a3ea8"}], nullBand)}
      ${L.shadow_frac ? railSVG("railShadow", "How much of that gap this 2D map can actually show (low = trust the map less; the rest is off-screen)", L.center, example.n_frames,
        [{vals: L.shadow_frac, color: "#3a8f5a"}]) : ""}
    </div>
    ${da ? `<p class="latentHint">Want to compare this clip's “nudge” to every other clip's? See the
      <button class="linkBtn" data-goto-map>Violation map</button> on the Population tab.</p>` : ""}
    ${metricGrid(example.metrics)}
    ${curationPanel(example)}`;
  initLatentPlayer(els.exampleDetail, example);
  initCurationPanel(els.exampleDetail, example);
  initViewSwitcher(els.exampleDetail);
  els.exampleDetail.querySelectorAll("[data-goto-map]").forEach((b) =>
    b.addEventListener("click", () => { state.popView = "map"; activateTab("population"); }));
}

function initLatentPlayer(root, example) {
  if (latentRAF) cancelAnimationFrame(latentRAF);
  const L = example.latent;
  const nF = example.n_frames;
  const fps = example.fps || 12;
  const pos = root.querySelector(".vPossible");
  const imp = root.querySelector(".vImpossible");
  const playBtn = root.querySelector(".rescorePlay");
  const scrub = root.querySelector(".rescoreScrub");
  const readout = root.querySelector(".rescoreReadout");
  const pTag = root.querySelector(".pTag");
  const iTag = root.querySelector(".iTag");
  const frameOf = (v) => Math.min(nF - 1, Math.max(0, Math.round(v.currentTime * fps)));

  // --- PCA panel: both trajectories in one space, with a moving dot per clip ---
  const panel = root.querySelector(".pcaPanel");
  const VB = 260, pad = 24;
  const xs = [...L.pca.possible, ...L.pca.impossible].map((p) => p[0]);
  const ysv = [...L.pca.possible, ...L.pca.impossible].map((p) => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ysv), ymax = Math.max(...ysv);
  const px = (x) => pad + ((x - xmin) / Math.max(1e-9, xmax - xmin)) * (VB - pad * 2);
  const py = (y) => VB - pad - ((y - ymin) / Math.max(1e-9, ymax - ymin)) * (VB - pad * 2);
  const path = (pts) => pts.map((p) => `${px(p[0])},${py(p[1])}`).join(" ");
  const dots = (pts, color) => pts.map((p) => `<circle cx="${px(p[0])}" cy="${py(p[1])}" r="1.6" fill="${color}" opacity="0.35" />`).join("");
  panel.innerHTML = `
    <polyline fill="none" stroke="#2364aa" stroke-width="1.5" opacity="0.5" points="${path(L.pca.possible)}" />
    <polyline fill="none" stroke="#bd3c3c" stroke-width="1.5" opacity="0.5" points="${path(L.pca.impossible)}" />
    ${dots(L.pca.possible, "#2364aa")}${dots(L.pca.impossible, "#bd3c3c")}
    <circle class="pcaDotP" r="5" fill="#2364aa" stroke="#fff" stroke-width="1.5" />
    <circle class="pcaDotI" r="5" fill="#bd3c3c" stroke="#fff" stroke-width="1.5" />`;
  const pcaDotP = panel.querySelector(".pcaDotP");
  const pcaDotI = panel.querySelector(".pcaDotI");

  const playLines = [...root.querySelectorAll(".railPlay")];
  const rails = [...root.querySelectorAll(".railChart")];
  const railX = (svg, f) => {
    const W = 600, padX = 40;
    return padX + (f / Math.max(1, nF - 1)) * (W - padX * 2);
  };
  const nearestWindow = (f) => {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < L.center.length; i += 1) {
      const d = Math.abs(L.center[i] - f);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  };

  // Localized divergence heatmap: the per-patch ‖pos − imp‖ for the nearest window,
  // painted on both clips (it's where they differ), reusing the surprise overlay.
  const dm = L.divergence_map || null;
  const pOvl = root.querySelector(".vPossibleOvl");
  const iOvl = root.querySelector(".vImpossibleOvl");
  const tgDiv = root.querySelector(".tgDiv");
  const drawDiv = (frame) => {
    if (!dm) return;
    const show = !!(tgDiv && tgDiv.checked);
    const grid = show ? dm.grid[nearestWindow(frame)] : null;
    drawOverlay(pOvl, pos, dm, grid, null, false, show);
    drawOverlay(iOvl, imp, dm, grid, null, false, show);
  };
  if (tgDiv) {
    tgDiv.addEventListener("change", () => drawDiv(Number(scrub.value)));
    pos.addEventListener("loadeddata", () => drawDiv(Number(scrub.value)));
    imp.addEventListener("loadeddata", () => drawDiv(Number(scrub.value)));
  }

  function place(frame) {
    const j = nearestWindow(frame);
    const pp = L.pca.possible[j], ip = L.pca.impossible[j];
    pcaDotP.setAttribute("cx", px(pp[0])); pcaDotP.setAttribute("cy", py(pp[1]));
    pcaDotI.setAttribute("cx", px(ip[0])); pcaDotI.setAttribute("cy", py(ip[1]));
    playLines.forEach((ln, i) => {
      const x = railX(rails[i], frame);
      ln.setAttribute("x1", x); ln.setAttribute("x2", x);
    });
    drawDiv(frame);
    const div = interpAt(L.center, L.divergence, frame);
    readout.textContent = `frame ${Math.round(frame)}/${nF - 1} · normal↔impossible gap ${div.toFixed(3)} · features-used ${L.eff_rank.impossible[j].toFixed(1)}`;
  }

  function loop() {
    if (!root.isConnected) return;
    const frame = pos.currentTime * fps;
    if (Math.abs(imp.currentTime - pos.currentTime) > 0.08) imp.currentTime = pos.currentTime;
    scrub.value = String(Math.min(nF - 1, Math.round(frame)));
    place(frame);
    pTag.textContent = `frame ${frameOf(pos)}`;
    iTag.textContent = `frame ${frameOf(imp)}`;
    latentRAF = requestAnimationFrame(loop);
  }
  function play() {
    pos.play(); imp.play();
    playBtn.textContent = "⏸ pause";
    latentRAF = requestAnimationFrame(loop);
  }
  function stop() {
    pos.pause(); imp.pause();
    if (latentRAF) cancelAnimationFrame(latentRAF);
    latentRAF = null;
    playBtn.textContent = "▶ play";
  }
  playBtn.addEventListener("click", () => (latentRAF ? stop() : play()));
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
  place(0);
  pTag.textContent = "frame 0";
  iTag.textContent = "frame 0";
}

// Shared lockstep player wiring: drives two <video>s + a scrub + a playhead,
// calling place(frame) each tick. Used by the anticipation view (the latent view
// has its own copy with extra overlay handling).
function syncLockstep({pos, imp, scrub, playBtn, fps, nF, place, root, tag}) {
  const frameOf = (v) => Math.min(nF - 1, Math.max(0, Math.round(v.currentTime * fps)));
  function loop() {
    if (!root.isConnected) return;
    const frame = pos.currentTime * fps;
    if (Math.abs(imp.currentTime - pos.currentTime) > 0.08) imp.currentTime = pos.currentTime;
    scrub.value = String(Math.min(nF - 1, Math.round(frame)));
    place(frame);
    if (tag) { tag.p.textContent = `frame ${frameOf(pos)}`; tag.i.textContent = `frame ${frameOf(imp)}`; }
    latentRAF = requestAnimationFrame(loop);
  }
  function play() { pos.play(); imp.play(); playBtn.textContent = "⏸ pause"; latentRAF = requestAnimationFrame(loop); }
  function stop() { pos.pause(); imp.pause(); if (latentRAF) cancelAnimationFrame(latentRAF); latentRAF = null; playBtn.textContent = "▶ play"; }
  playBtn.addEventListener("click", () => (latentRAF ? stop() : play()));
  scrub.addEventListener("input", () => {
    const f = Number(scrub.value), t = f / fps;
    pos.currentTime = t; imp.currentTime = t; place(f);
    if (tag) { tag.p.textContent = `frame ${f}`; tag.i.textContent = `frame ${f}`; }
  });
  pos.addEventListener("ended", () => { stop(); pos.currentTime = 0; imp.currentTime = 0; place(0); scrub.value = "0"; });
  place(0);
  if (tag) { tag.p.textContent = "frame 0"; tag.i.textContent = "frame 0"; }
}

// --- multi-view explorer: a switcher over the lenses a latent run carries ---

// Per-clip lenses only — these change as you pick a different pair. The two
// population read-outs (Probes, Violation map) are identical for every clip, so
// they live on the dedicated Population tab, not in this per-clip switcher.
const VIEW_LABELS = {latent: "Latent surface", anticipation: "Anticipation", dense: "Dense features"};

function availableViews(example) {
  const views = ["latent"];
  if (example.latent?.anticipation) views.push("anticipation");
  if (example.latent?.dense_pca) views.push("dense");
  return views;
}

function setExploreView(v) {
  state.exploreView = v;
  renderExampleDetail();
}

function viewSwitcher(active, example) {
  const views = availableViews(example);
  if (views.length < 2) return "";
  return `<div class="viewSwitch">${views.map((v) =>
    `<button class="viewTab${v === active ? " is-active" : ""}" data-view="${v}">${VIEW_LABELS[v]}</button>`).join("")}</div>`;
}

function initViewSwitcher(root) {
  root.querySelectorAll(".viewTab").forEach((b) =>
    b.addEventListener("click", () => setExploreView(b.dataset.view)));
}

function exploreHead(example, viewName) {
  const above = example.label === "above null";
  return `<div class="rescoreHead">
      <h2>${example.id} <span class="badge ${above ? "badgeGood" : "badgeBad"}">${example.label || "latent"}</span>
        <span class="viewName">· ${viewName}</span></h2>
    </div>
    ${viewSwitcher(state.exploreView, example)}`;
}

// --- Anticipation view: the predictor's output vs the actual future, per clip.
// Surprise is just the magnitude of this gap; here we show its trajectory. ---
function predActualSVG(side, ant, color) {
  const VB = 260, pad = 24;
  const all = [...ant.pred_xy, ...ant.actual_xy];
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (x) => pad + ((x - xmin) / ((xmax - xmin) || 1)) * (VB - 2 * pad);
  const py = (y) => VB - pad - ((y - ymin) / ((ymax - ymin) || 1)) * (VB - 2 * pad);
  const path = (pts, dash) => `<polyline fill="none" stroke="${color}" stroke-width="1.6" opacity="0.6"${dash ? ` stroke-dasharray="4 3"` : ""} points="${pts.map((p) => `${px(p[0])},${py(p[1])}`).join(" ")}" />`;
  const off = ant.offset != null ? ` · constant predictor↔target offset of ${ant.offset.toFixed(1)} removed` : "";
  const cm = ant.comove != null
    ? ` · <strong>move-together r = ${ant.comove.toFixed(2)}</strong> (1 = the prediction anticipates every turn; 0 = unrelated)`
    : "";
  return `<figure class="chartCard">
    <figcaption><strong style="color:${color}">${side}</strong> — predicted (dashed) vs actual (solid), aligned to a shared center so the paths start together${off}; where they pull apart, the model genuinely mispredicted that moment${cm}</figcaption>
    <svg class="antPanel" viewBox="0 0 ${VB} ${VB}">
      ${path(ant.actual_xy, false)}${path(ant.pred_xy, true)}
      <circle class="antActual" r="5" fill="${color}" stroke="#fff" stroke-width="1.5"/>
      <circle class="antPred" r="5" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="2 2"/></svg></figure>`;
}

function renderAnticipationView(example) {
  const A = example.latent.anticipation;
  els.exampleDetail.className = "";
  els.exampleDetail.innerHTML = `
    ${exploreHead(example, "Anticipation")}
    <p class="latentHint"><strong>In plain terms:</strong> part of V-JEPA tries to <em>predict what
      happens next</em>. The <strong>solid</strong> line is what actually happened (in the model's own
      terms); the <strong>dashed</strong> line is what it predicted. The predictor and the target are
      two different networks, so their outputs sit a near-constant distance apart every frame — an offset
      that says nothing about <em>when</em> the model is surprised. We subtract that constant so the two
      paths start together; what's left is the real story: where they pull apart, the prediction genuinely
      drifted off the future. (The raw gap, offset included, is the “surprise” rails below.)</p>
    <div class="rescoreControls">
      <button class="rescorePlay">▶ play</button>
      <input class="rescoreScrub" type="range" min="0" max="${example.n_frames - 1}" value="0" step="1" />
      <span class="rescoreReadout"></span>
    </div>
    <div class="rescoreVideos latentVideos">
      ${videoFigure("Possible", "vPossible", "pTag", assetUrl(example.video_possible))}
      ${videoFigure("Impossible", "vImpossible", "iTag", assetUrl(example.video_impossible))}
    </div>
    <div class="antPanels">
      ${predActualSVG("Possible", A.possible, "#2364aa")}
      ${predActualSVG("Impossible", A.impossible, "#bd3c3c")}
    </div>
    <div class="rails">
      ${railSVG("antErrP", "How wrong its prediction was, moment by moment — normal clip", example.latent.center, example.n_frames, [{vals: A.possible.err, color: "#2364aa"}])}
      ${railSVG("antErrI", "How wrong its prediction was, moment by moment — impossible clip", example.latent.center, example.n_frames, [{vals: A.impossible.err, color: "#bd3c3c"}])}
    </div>`;
  initAnticipationView(els.exampleDetail, example);
  initViewSwitcher(els.exampleDetail);
}

function initAnticipationView(root, example) {
  if (latentRAF) cancelAnimationFrame(latentRAF);
  const A = example.latent.anticipation, C = example.latent.center;
  const nF = example.n_frames, fps = example.fps || 12;
  const pos = root.querySelector(".vPossible"), imp = root.querySelector(".vImpossible");
  const scrub = root.querySelector(".rescoreScrub"), playBtn = root.querySelector(".rescorePlay");
  const readout = root.querySelector(".rescoreReadout");
  const panels = [...root.querySelectorAll(".antPanel")];
  const playLines = [...root.querySelectorAll(".railPlay")];
  const rails = [...root.querySelectorAll(".railChart")];
  const sides = [{p: panels[0], a: A.possible}, {p: panels[1], a: A.impossible}];
  const nearest = (f) => { let b = 0, bd = Infinity; C.forEach((c, i) => { const d = Math.abs(c - f); if (d < bd) { bd = d; b = i; } }); return b; };
  const railX = (f) => 40 + (f / Math.max(1, nF - 1)) * (600 - 80);
  // recompute panel coords (same scaling as predActualSVG) to place the dots
  const placePanel = ({p, a}, j) => {
    const VB = 260, pad = 24, all = [...a.pred_xy, ...a.actual_xy];
    const xs = all.map((q) => q[0]), ys = all.map((q) => q[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const px = (x) => pad + ((x - xmin) / ((xmax - xmin) || 1)) * (VB - 2 * pad);
    const py = (y) => VB - pad - ((y - ymin) / ((ymax - ymin) || 1)) * (VB - 2 * pad);
    const act = p.querySelector(".antActual"), prd = p.querySelector(".antPred");
    act.setAttribute("cx", px(a.actual_xy[j][0])); act.setAttribute("cy", py(a.actual_xy[j][1]));
    prd.setAttribute("cx", px(a.pred_xy[j][0])); prd.setAttribute("cy", py(a.pred_xy[j][1]));
  };
  const place = (frame) => {
    const j = nearest(frame);
    sides.forEach((s) => placePanel(s, j));
    playLines.forEach((ln) => { const x = railX(frame); ln.setAttribute("x1", x); ln.setAttribute("x2", x); });
    readout.textContent = `frame ${Math.round(frame)}/${nF - 1} · how wrong the prediction was — normal ${A.possible.err[j].toFixed(2)} · impossible ${A.impossible.err[j].toFixed(2)}`;
  };
  syncLockstep({pos, imp, scrub, playBtn, fps, nF, place, root,
    tag: {p: root.querySelector(".pTag"), i: root.querySelector(".iTag")}});
}

// --- Dense view: per-clip top-3 PCA of the patch grid as an RGB segmentation. ---
function densePcaSVG(grid, label) {
  const H = grid.length, W = grid[0].length, cell = 18;
  let rects = "";
  for (let r = 0; r < H; r += 1) for (let c = 0; c < W; c += 1) {
    const [R, G, B] = grid[r][c].map((v) => Math.round(v * 255));
    rects += `<rect x="${c * cell}" y="${r * cell}" width="${cell}" height="${cell}" fill="rgb(${R},${G},${B})"/>`;
  }
  return `<figure class="denseFig"><figcaption>${label}</figcaption>
    <svg class="denseGrid" viewBox="0 0 ${W * cell} ${H * cell}">${rects}</svg></figure>`;
}

function renderDenseView(example) {
  const D = example.latent.dense_pca;
  els.exampleDetail.className = "";
  els.exampleDetail.innerHTML = `
    ${exploreHead(example, "Dense features")}
    <p class="latentHint"><strong>In plain terms:</strong> we color each square of the frame by what
      the model “sees” there — squares it represents similarly get similar colors. It's a rough map of
      how the model carves up the scene (does it pick out the object, the wall, the floor?). Colors are
      arbitrary and only meaningful <em>within</em> one clip, not between the two.
      <strong>This is one static map per clip — averaged over the whole clip, not a single frame</strong>
      — so there's nothing to scrub or play; it answers “how does the model partition this scene on
      average,” not “how does it change frame to frame.”</p>
    <div class="rescoreVideos latentVideos">
      ${videoFigure("Possible", "vPossible", "pTag", assetUrl(example.video_possible))}
      ${videoFigure("Impossible", "vImpossible", "iTag", assetUrl(example.video_impossible))}
    </div>
    <div class="densePanels">
      ${densePcaSVG(D.possible, "Possible clip")}
      ${densePcaSVG(D.impossible, "Impossible clip")}
    </div>`;
  initViewSwitcher(els.exampleDetail);
}

// --- Probes view (population): is a factor linearly decodable from the frozen
// latent, vs a label-shuffle null; and at what depth does it emerge? ---
function probeBar(f) {
  // label row on top, the 0..1 accuracy track below it (with the shuffle-null tick).
  const W = 480, H = 42, x0 = 10, x1 = W - 130, trackY = 30;
  const sx = (v) => x0 + v * (x1 - x0);
  return `<svg class="probeBar" viewBox="0 0 ${W} ${H}">
    <text x="${x0}" y="13" class="probeLbl">${f.name}</text>
    <line x1="${x0}" y1="${trackY}" x2="${x1}" y2="${trackY}" stroke="#e1e5ea"/>
    <rect x="${x0}" y="${trackY - 4}" width="${sx(f.acc) - x0}" height="8" rx="2" fill="${f.acc - f.null > 0.1 ? "#2f9e57" : "#bd3c3c"}"/>
    <rect x="${sx(f.null)}" y="${trackY - 7}" width="2" height="14" fill="#444"><title>shuffle null ${f.null}</title></rect>
    <text x="${x1 + 8}" y="${trackY + 4}" class="probeVal">${(f.acc * 100).toFixed(0)}% · null ${(f.null * 100).toFixed(0)}%</text>
  </svg>`;
}

function layerwiseSVG(lw) {
  const W = 600, H = 200, pad = 40;
  const n = lw.layers.length;
  const sx = (i) => pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
  const sy = (v) => H - pad - v * (H - 2 * pad);
  const poly = (vals, color, dash) => `<polyline fill="none" stroke="${color}" stroke-width="2"${dash ? ` stroke-dasharray="4 3"` : ""} points="${vals.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")}"/>`;
  return `<svg class="layerwise" viewBox="0 0 ${W} ${H}">
    <line x1="${pad}" y1="${sy(0.5)}" x2="${W - pad}" y2="${sy(0.5)}" stroke="#e1e5ea" stroke-dasharray="3 3"/>
    <text x="${W - pad}" y="${sy(0.5) - 5}" text-anchor="end" class="svgAxis">guessing (50%)</text>
    ${poly(lw.acc, "#2364aa", false)}${poly(lw.null, "#888", true)}
    <text x="${pad}" y="${H - 8}" class="svgAxis">shallow (layer 0)</text>
    <text x="${W - pad}" y="${H - 8}" text-anchor="end" class="svgAxis">deep (layer ${n - 1}) →</text>
  </svg>`;
}

// --- Violation map (population): every clip's impossible-event "nudge" as a map +
// a similarity grid, with a live clip preview so a cluster leads to the footage. ---
function renderMapPanel(host) {
  const da = state.manifest?.latent_space?.delta_analysis;
  if (!da) {
    host.innerHTML = popHead("Violation map") + popSwitcher("map") + `<div class="empty">No violation-direction analysis in this run.</div>`;
    return;
  }
  // Focus on the clip you last looked at in Explore, else the first one.
  const focus = (state.selectedExampleId && da.specs.includes(state.selectedExampleId)) ? state.selectedExampleId : da.specs[0];
  host.innerHTML = `
    ${popHead("Violation map")}
    ${popSwitcher("map")}
    <p class="latentHint"><strong>In plain terms:</strong> for each clip we take the difference between
      how the model reads the impossible version vs its possible twin, averaged over the <em>whole</em>
      clip — one “nudge” direction per clip (not per frame or segment). This compares those nudges across
      all ${da.specs.length} clips: each dot / each row-and-column is one full clip, and clips nudged the
      same way land near each other on the map (and show red on the grid).
      <strong>Hover a dot or square to preview the clip(s); click to pin them so you can compare
      without the preview chasing your cursor; “open ▶” jumps to that clip's full view in Explore.</strong></p>
    <div class="dirLayout">
      ${deltaScatterSVG(da, focus)}
      <div class="cosWrap">
        <div class="chartCap">Similarity of two clips' nudges — hover a square to compare, click to pin</div>
        ${cosMatrixSVG(da, focus)}
        <div class="cosReadout">Hover a square: red = the two clips' impossible events move the model the same way, blue = opposite, white = unrelated.</div>
        ${cosBarSVG(da.null_cos95)}
      </div>
    </div>
    <div class="dirPreview"><div class="dirPinStatus"></div><div class="previewCards"></div></div>`;
  initDirectionsView(host, {id: focus}, da);
}

function initDirectionsView(root, example, da) {
  const cards = root.querySelector(".previewCards");
  const status = root.querySelector(".dirPinStatus");
  const exBySpec = (s) => state.examples.find((e) => e.id === s);
  const openPair = (spec) => {
    if (!spec) return;
    state.selectedExampleId = spec;
    state.exploreView = "latent";
    activateTab("examples");          // the map lives on Population; the clip opens in Explore
    renderExampleList();
    renderExampleDetail();
  };

  // `pinned` non-null means the user clicked to lock a comparison in place.
  // While pinned, hover is ignored so the preview (and the layout) stop moving —
  // that's what lets you study two clips without the rerender chasing your cursor.
  let pinned = null;
  let lastKey = "";
  const draw = (specs, isPinned) => {
    const key = (isPinned ? "P:" : "H:") + specs.join("|");
    if (key === lastKey) return;   // don't rebuild <video>s on every mousemove
    lastKey = key;
    cards.innerHTML = specs.map((s) => {
      const ex = exBySpec(s);
      if (!ex) return "";
      return `<figure class="previewCard${isPinned ? " pinned" : ""}">
        <figcaption>${s} — impossible clip <button class="linkBtn" data-open="${s}">open ▶</button></figcaption>
        <video muted loop autoplay playsinline src="${assetUrl(ex.video_impossible)}"></video></figure>`;
    }).join("");
    cards.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => openPair(b.dataset.open)));
  };
  const setStatus = () => {
    if (pinned) {
      status.innerHTML = `📌 Pinned <strong>${pinned.join("</strong> vs <strong>")}</strong> — `
        + `move freely, then <button class="linkBtn dirClear">clear</button> to follow the cursor again.`;
      status.querySelector(".dirClear").addEventListener("click", () => {
        pinned = null; draw([example.id], false); litDot([example.id]); setStatus();
      });
    } else {
      status.textContent = "Hover the map to preview; click a dot or square to pin it here so it stops moving.";
    }
  };
  // light up the matching dot(s) in the scatter so a heatmap pick is visible there too
  const dots = [...root.querySelectorAll(".dScatterDot")];
  const litDot = (specs) => {
    const set = new Set(specs);
    dots.forEach((d) => {
      const on = set.has(d.dataset.spec);
      d.setAttribute("r", on ? 7 : 4);
      d.setAttribute("stroke", on ? "#111" : "#fff");
      d.setAttribute("stroke-width", on ? 2.5 : 1);
      if (on) d.parentNode.appendChild(d);   // raise lit dots to the front
    });
  };
  const hover = (specs) => { if (!pinned) { draw(specs, false); litDot(specs); } };
  const pin = (specs) => { pinned = specs; lastKey = ""; draw(specs, true); litDot(specs); setStatus(); };

  root.querySelectorAll(".dScatterDot").forEach((dot) => {
    dot.addEventListener("mouseenter", () => hover([dot.dataset.spec]));
    dot.addEventListener("click", (e) => { e.stopPropagation(); pin([dot.dataset.spec]); });
  });

  const matrix = root.querySelector(".cosMatrix");
  if (matrix) {
    const overlay = matrix.querySelector(".cosOverlay");
    const readout = root.querySelector(".cosReadout");
    const mark = root.querySelector(".cosBar .cosMark");
    const N = da.specs.length;
    const cellOf = (e) => {
      const r = overlay.getBoundingClientRect();
      const j = Math.max(0, Math.min(N - 1, Math.floor(((e.clientX - r.left) / r.width) * N)));
      const i = Math.max(0, Math.min(N - 1, Math.floor(((e.clientY - r.top) / r.height) * N)));
      return [i, j];
    };
    const specsFor = (i, j) => (i === j ? [da.specs[i]] : [da.specs[i], da.specs[j]]);
    overlay.addEventListener("mousemove", (e) => {
      const [i, j] = cellOf(e);
      const c = da.cosine_matrix[i][j];
      const verdict = Math.abs(c) <= da.null_cos95 ? "unrelated (within noise)"
        : c > 0 ? "same way" : "opposite";
      readout.textContent = `${da.specs[i]}  vs  ${da.specs[j]} · ${c >= 0 ? "+" : ""}${c.toFixed(2)} · ${verdict}`;
      if (mark) { const x = 10 + ((c + 1) / 2) * (300 - 20); mark.setAttribute("x1", x); mark.setAttribute("x2", x); }
      hover(specsFor(i, j));
    });
    overlay.addEventListener("click", (e) => { const [i, j] = cellOf(e); pin(specsFor(i, j)); });
  }

  draw([example.id], false);   // default: the pair you came in on
  litDot([example.id]);
  setStatus();
}

function renderProbesPanel(host) {
  const p = state.manifest?.latent_space?.probes;
  if (!p) {
    host.innerHTML = popHead("Probes") + popSwitcher("probes") + `<div class="empty">No probe analysis in this run.</div>`;
    return;
  }
  const partial = p.validity_motion_partialled;
  // Lead with the plain verdict computed from the data, not jargon.
  const find = (sub) => (p.factors || []).find((f) => f.name.toLowerCase().includes(sub));
  const phys = find("impossible");
  const scene = find("scene");
  const pct = (v) => `${Math.round(v * 100)}%`;
  const readable = phys && phys.acc - phys.null > 0.1;
  const verdict = phys
    ? `<p class="probeVerdict ${readable ? "vGood" : "vBad"}">
        The model's frozen features <strong>${readable ? "can" : "cannot"}</strong> tell impossible from
        normal: <strong>${pct(phys.acc)}</strong> vs ${pct(phys.null)} for pure guessing${readable ? "" : " — i.e. at chance"}.
        ${scene ? `Yet the same features read off <em>which scene type</em> it is at <strong>${pct(scene.acc)}</strong>,
        and predict <em>how much is moving</em> almost perfectly` : ""}${typeof p.motion_r2 === "number" ? ` (r² = ${p.motion_r2.toFixed(2)})` : ""}.
        ${readable ? "" : "So the representation richly encodes scene and motion, but the <strong>“did physics break?” signal isn't linearly present at all</strong> — the identifiability version of the headline surprise result."}
      </p>`
    : "";
  host.innerHTML = `
    ${popHead("Probes")}
    ${popSwitcher("probes")}
    ${verdict}
    <p class="latentHint"><strong>How to read this:</strong> we freeze the model's features and hand a
      <em>simple</em> classifier one fact to read off (“is this impossible?”, “which scene type?”,
      “how much motion?”), scored against a <strong>shuffled-label baseline</strong> (pure guessing).
      <strong>A bar past the dark tick = a fact the model genuinely represents; a bar sitting at the tick
      = it can't tell.</strong> This summarizes the whole set, so it's the same for every clip.</p>
    <div class="probeBlock">
      <h3>Can the model's features tell these apart?</h3>
      ${(p.factors || []).map(probeBar).join("") || `<div class="empty">No probe factors for this run.</div>`}
      ${partial ? `<p class="probeNote"><strong>The key control:</strong> can it still tell normal from impossible <em>after we erase the “how much is moving” signal</em> from the features? <strong>${pct(partial.acc)}</strong> (guessing ≈ ${pct(partial.null)}). At the guessing level means the little it had was just motion, not physics.</p>` : ""}
    </div>
    ${p.layerwise ? `<div class="probeBlock"><h3>Where in the network does this show up?</h3>
      <p class="probeNote">Reading “impossible vs normal” at each layer, shallow (left) to deep (right). Blue rising above the grey baseline = where the model first represents it.</p>
      ${layerwiseSVG(p.layerwise)}</div>` : ""}`;
}

// --- Population tab: the read-outs that are the same for every clip ----------
const POP_LABELS = {probes: "Probes", map: "Violation map"};

function availablePopViews() {
  const ls = state.manifest?.latent_space;
  const v = [];
  if (ls?.probes) v.push("probes");
  if (ls?.delta_analysis) v.push("map");
  return v;
}

function popHead(viewName) {
  return `<div class="rescoreHead"><h2>Population <span class="viewName">· ${viewName}</span>
    <span class="popSub">— summarizes all ${state.examples.length} clips; the same for every pair</span></h2></div>`;
}

function popSwitcher(active) {
  const views = availablePopViews();
  if (views.length < 2) return "";
  return `<div class="viewSwitch">${views.map((v) =>
    `<button class="viewTab${v === active ? " is-active" : ""}" data-popview="${v}">${POP_LABELS[v]}</button>`).join("")}</div>`;
}

function setPopView(v) {
  state.popView = v;
  renderPopulation();
}

function renderPopulation() {
  const host = els.populationDetail;
  if (!host) return;
  const views = availablePopViews();
  if (!views.length) {
    host.className = "popDetail empty";
    host.textContent = "This run has no population-level analysis (probes / violation map).";
    return;
  }
  if (!views.includes(state.popView)) state.popView = views[0];
  host.className = "popDetail";
  if (state.popView === "map") renderMapPanel(host);
  else renderProbesPanel(host);
  host.querySelectorAll("[data-popview]").forEach((b) =>
    b.addEventListener("click", () => setPopView(b.dataset.popview)));
}

function renderExampleDetail() {
  const example = state.examples.find((item) => item.id === state.selectedExampleId);
  if (!example) {
    els.exampleDetail.className = "empty";
    els.exampleDetail.textContent = state.examples.length ? "Select a pair." : "No pairs in this run.";
    return;
  }
  if (example.latent && example.video_possible) {
    const views = availableViews(example);
    if (!views.includes(state.exploreView)) state.exploreView = "latent";
    ({
      latent: renderLatentPlayer,
      anticipation: renderAnticipationView,
      dense: renderDenseView,
    }[state.exploreView] || renderLatentPlayer)(example);
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

// Surprise-run summary helper (the VoE chips on Home); latent runs ignore it.
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
  // Latent-surface runs carry no surprise_gap; summarize the latent read instead.
  if (state.examples.some((e) => e.latent)) {
    const lat = state.examples.filter((e) => e.latent);
    const above = lat.filter((e) => e.label === "above null").length;
    const ranks = lat.map((e) => e.metrics?.min_eff_rank_imp).filter((v) => typeof v === "number");
    const medRank = ranks.length ? ranks.slice().sort((a, b) => a - b)[Math.floor(ranks.length / 2)] : null;
    els.homeStats.innerHTML = `
      <span class="statChip">Current run: <strong>${run.id || "—"}</strong> · latent surface</span>
      <span class="statChip">${lat.length} pairs</span>
      <span class="statChip"><strong>${above}/${lat.length}</strong> exceed within-scene null</span>
      ${medRank !== null ? `<span class="statChip">median min effective rank <strong>${medRank.toFixed(1)}</strong> / 1024</span>` : ""}`;
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
  if (name === "population") renderPopulation();   // population content is built on demand
}

document.querySelector(".tabs").addEventListener("click", (event) => {
  const button = event.target.closest(".tab");
  if (!button) return;
  activateTab(button.dataset.tab);
});

els.runSelect.addEventListener("change", () => loadManifest(els.runSelect.value));
els.exampleFilter.addEventListener("input", renderExampleList);
els.metricSort.addEventListener("change", renderExampleList);
els.exampleList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  state.selectedExampleId = button.dataset.example;
  renderExampleList();
  renderExampleDetail();
});

// Deep-link the active tab via the URL hash (#examples) so a view is shareable
// and reload-stable.
const TAB_NAMES = ["home", "examples", "population"];
// `#examples`, `#examples:dense`, or `#population:map` — optionally deep-links a view.
function tabFromHash() {
  const [name, view] = (location.hash || "").replace(/^#/, "").split(":");
  // Legacy links: probes/map used to live under #examples; they're now Population.
  if (name === "examples" && (view === "probes" || view === "map")) {
    state.popView = view;
    activateTab("population");
    return;
  }
  if (name === "population") {
    if (view === "probes" || view === "map") state.popView = view;
    activateTab("population");
    return;
  }
  if (TAB_NAMES.includes(name)) activateTab(name);
  if (view && view !== state.exploreView) {
    state.exploreView = view;
    renderExampleDetail();
  }
}
window.addEventListener("hashchange", tabFromHash);

loadRuns()
  .then(() => loadManifest(state.runId))
  .then(tabFromHash)
  .catch((error) => {
    els.runMeta.textContent = error.message;
  });
