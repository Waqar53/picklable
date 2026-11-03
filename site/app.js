// PickLabel live demo: UI on the main thread, the Python pipeline in a Web Worker (Pyodide).
const $ = (s) => document.querySelector(s);

const ROUTES = {
  accept: ["ACCEPT · ground truth", "Confident: uploaded to Roboflow as a ground-truth annotation."],
  review: ["REVIEW · human check", "Plausible: uploaded as a prediction into Roboflow's review queue, out of the dataset until approved."],
  hard: ["HARD · label first", "Failed or ambiguous pick: uploaded tagged priority:high so a human labels it first. These are where perception was wrong."],
};
const REASONS = {
  no_change_at_grasp: "nothing left the scene at the grasp point",
  gripper_empty: "gripper reported empty",
  gripper_holding_but_no_visible_change: "gripper holding, but no visible change (clear film?)",
  gripper_empty_but_scene_changed: "gripper empty but the scene changed: pushed, not picked",
  grasp_outside_mask: "grasp point outside the mask",
  grasp_on_edge: "grasp on the item's edge",
  grasp_not_visible_after: "the item left the after frame",
  alignment_suspect: "frame alignment is suspect",
  weak_change: "weak change signal",
  mask_disagrees_with_change: "mask disagrees with the change region",
  implausible_shape: "implausible shape",
};
const SOURCES = { change: "change map + GrabCut", sam2: "SAM 2.1", sam3: "SAM 3", none: "none" };

const state = {
  examples: [], key: null, grasp: null, graspSuccess: null, gt: false,
  ready: false, loaded: new Set(), info: {}, busy: false, pending: null, runId: 0, precomputed: {},
};

const worker = new Worker("worker.js");
worker.onmessage = ({ data }) => {
  if (data.type === "progress") setStatus(data.text + " (first visit only)");
  else if (data.type === "ready") { state.ready = true; setStatus("Live: Python + OpenCV running in your browser. Click the before frame to move the grasp point.", "ready"); maybeRun(); }
  else if (data.type === "fatal") setStatus("Live mode unavailable in this browser (" + data.text + "). Showing precomputed results.", "error");
  else if (data.type === "loaded") { state.loaded.add(data.key); state.info[data.key] = data.info; if (data.key === state.key) maybeRun(); }
  else if (data.type === "result") {
    state.busy = false;
    if (data.key === state.key) render(data.result, true);
    if (state.pending) { const p = state.pending; state.pending = null; send(p); } else setRunning(false);
  } else if (data.type === "error") {
    state.busy = false; setRunning(false);
    setStatus("Error: " + data.text, "error");
  }
};

function setStatus(text, cls = "") {
  $("#status-text").textContent = text;
  $("#status").className = "status " + cls;
}
function setRunning(on) { document.querySelector(".stage").classList.toggle("running", on); }

async function bytes(url) { return new Uint8Array(await (await fetch(url)).arrayBuffer()); }

function placeMarker() {
  const img = $("#img-before");
  if (!state.grasp || !img.naturalWidth) return;
  const m = $("#marker");
  m.hidden = false;
  m.style.left = (100 * state.grasp[0] / img.naturalWidth) + "%";
  m.style.top = (100 * state.grasp[1] / img.naturalHeight) + "%";
}

async function selectExample(ex) {
  state.key = ex.key; state.grasp = ex.grasp.slice(); state.graspSuccess = ex.grasp_success; state.gt = true;
  document.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", String(t.dataset.key === ex.key)));
  const img = $("#img-before");
  img.onload = placeMarker;
  img.src = ex.before;
  if (state.precomputed[ex.key]) render(state.precomputed[ex.key], false);
  else {
    fetch(ex.result).then((r) => r.json()).then((res) => {
      state.precomputed[ex.key] = res;
      if (state.key === ex.key && !state.ready) render(res, false);
    });
  }
  if (!state.loaded.has(ex.key)) {
    const [b, a, g] = await Promise.all([bytes(ex.before), bytes(ex.after), bytes(ex.gt)]);
    worker.postMessage({ type: "load", key: ex.key, before: b, after: a, gt: g }, [b.buffer, a.buffer, g.buffer]);
  } else maybeRun();
}

function maybeRun() {
  if (!state.ready || !state.loaded.has(state.key) || !state.grasp) return;
  const scale = (state.info[state.key] || {}).scale || 1;
  const req = { type: "run", id: ++state.runId, key: state.key, gx: state.grasp[0] * scale, gy: state.grasp[1] * scale,
                graspSuccess: state.graspSuccess };
  if (state.busy) { state.pending = req; return; }
  send(req);
}
function send(req) { state.busy = true; setRunning(true); worker.postMessage(req); }

function render(r, live) {
  $("#img-after").src = r.images.after;
  $("#img-change").src = r.images.change;
  $("#img-result").src = r.images.result;
  const [name, explain] = ROUTES[r.route];
  const route = $("#route");
  route.textContent = name;
  route.className = "route " + r.route;
  $("#route-explain").textContent = explain;
  $("#score").textContent = r.score.toFixed(2);
  $("#iou").textContent = r.iou === null || r.iou === undefined ? "–" : r.iou.toFixed(2);
  $("#iou").title = r.iou === null ? "Shown when there is a mask and the grasp is on the item that was actually picked" : "";
  $("#source").textContent = SOURCES[r.source] || r.source;
  $("#ms").textContent = live ? (r.ms / 1000).toFixed(1) + " s here" : "precomputed";
  $("#reasons").innerHTML = "";
  for (const code of r.reasons) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = REASONS[code] || code;
    $("#reasons").appendChild(c);
  }
  const sig = $("#signals");
  sig.innerHTML = "";
  for (const [k, v] of Object.entries(r.signals)) {
    const row = document.createElement("div");
    row.className = "sig";
    row.innerHTML = `<span>${k}</span><span class="bar"><i style="width:${Math.round(100 * v)}%"></i></span><b>${v.toFixed(2)}</b>`;
    sig.appendChild(row);
  }
  if (!Object.keys(r.signals).length) sig.innerHTML = '<p class="fine">No mask, so no signals: the pick was routed on its own.</p>';
  const a = r.align || {};
  const gains = r.gains ? " · white-balance gains " + r.gains.map((g) => g.toFixed(2)).join(" / ") : "";
  $("#align-info").textContent = `Alignment: ${a.method} (residual ${a.residual})${gains}`;
  $("#payload").textContent = JSON.stringify(r.payload, null, 2);
}

$("#before-frame").addEventListener("click", (e) => {
  const img = $("#img-before");
  if (!img.naturalWidth) return;
  const rect = img.getBoundingClientRect();
  // the image is letterboxed inside the frame (object-fit: contain)
  const s = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
  const ox = (rect.width - img.naturalWidth * s) / 2, oy = (rect.height - img.naturalHeight * s) / 2;
  const x = (e.clientX - rect.left - ox) / s, y = (e.clientY - rect.top - oy) / s;
  if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) return;
  state.grasp = [x, y];
  state.graspSuccess = null;
  placeMarker();
  if (!state.ready) setStatus("Python is still loading; your click will run as soon as it is ready.");
  maybeRun();
});

const uploads = { before: null, after: null };
for (const which of ["before", "after"]) {
  $("#up-" + which).addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    uploads[which] = { bytes: new Uint8Array(await f.arrayBuffer()), url: URL.createObjectURL(f) };
    if (!(uploads.before && uploads.after)) return;
    const key = "upload-" + Date.now();
    state.key = key; state.graspSuccess = null; state.gt = false; state.grasp = null;
    document.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
    const img = $("#img-before");
    img.onload = () => { state.grasp = [img.naturalWidth / 2, img.naturalHeight / 2]; placeMarker(); maybeRun(); };
    img.src = uploads.before.url;
    worker.postMessage({ type: "load", key, before: uploads.before.bytes.slice(), after: uploads.after.bytes.slice() });
    setStatus("Click the item that was removed, on the before frame.", state.ready ? "ready" : "");
  });
}

window.addEventListener("resize", placeMarker);

(async () => {
  state.examples = await (await fetch("examples/examples.json")).json();
  const thumbs = $("#thumbs");
  for (const ex of state.examples) {
    const b = document.createElement("button");
    b.className = "thumb";
    b.dataset.key = ex.key;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<img src="${ex.before}" alt=""><span>${ex.label}<small>${ex.note}</small></span>`;
    b.addEventListener("click", () => selectExample(ex));
    thumbs.appendChild(b);
  }
  selectExample(state.examples[0]);
})();
