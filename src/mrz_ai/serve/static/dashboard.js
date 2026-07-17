/* Polls /api/status and draws it. Holds no state of its own: the file on disk is
 * the state, and a reader that reconstructs its own can be wrong about what is
 * happening — which is the one thing a monitor must not be.
 *
 * Nothing here can affect the run. Worth saying out loud, because the temptation
 * to add a "stop" button to a page like this is exactly how a dashboard ends up
 * able to kill forty minutes of GPU time by a misclick over a rented pod.
 */

const $ = (id) => document.getElementById(id);

/* Slow. There is nothing to see between polls -- the trainer writes every 100
 * steps -- and a tighter loop would only spend the pod's CPU on serialising JSON
 * for a browser nobody is looking at. */
const POLL_MS = 3000;

const fmt = {
  int: (n) => (n ?? 0).toLocaleString("en-US"),
  pct: (n) => (n == null ? "—" : `${(n * 100).toFixed(2)}%`),
  fixed: (n, d) => (n == null ? "—" : n.toFixed(d)),
  exp: (n) => (n == null ? "—" : n.toExponential(2)),
  /** Durations a person reads at a glance: "12m 30s", not "750.4s". */
  clock: (seconds) => {
    if (seconds == null || !isFinite(seconds)) return "—";
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
  },
};

const set = (id, text) => ($(id).textContent = text);

/** Bars from divs, scaled from the foot. `values` are already 0..1. */
function chart(element, values, floor = 0) {
  element.replaceChildren();
  if (!values.length) {
    const empty = document.createElement("span");
    empty.className = "empty";
    empty.textContent = "No evaluation yet";
    element.append(empty);
    return;
  }
  // Rescaled from `floor` rather than from zero. Accuracy lives in its last few
  // percent — every bar pinned to the ceiling shows a run that has stopped
  // improving exactly like one that is improving, which is the question being
  // asked.
  for (const value of values) {
    const bar = document.createElement("div");
    bar.className = "bar";
    const scaled = Math.max(0, Math.min(1, (value - floor) / (1 - floor)));
    bar.style.transform = `scaleY(${Math.max(scaled, 0.01)})`;
    element.append(bar);
  }
}

function render(body) {
  $("watching").textContent = body.watching ?? "—";

  const state = body.stale ? "stale" : body.state ?? "waiting";
  const dot = $("run-dot");
  dot.className =
    state === "training" ? "dot ok" : state === "finished" ? "dot ok" : "dot bad";
  set(
    "run-state",
    { training: "Training", finished: "Finished", failed: "Failed", stale: "No signal", waiting: "Waiting for the run" }[
      state
    ] ?? state
  );

  $("stale").hidden = !body.stale;
  if (body.stale) set("stale-age", fmt.clock(body.age_seconds));

  $("failed").hidden = body.state !== "failed";
  if (body.state === "failed") set("failed-why", body.error ?? "");

  // The one number on this page worth interrupting somebody over. A run on a
  // single cut of OCR-B is a run to throw away, and its loss curve looks perfect.
  const fonts = body.fonts ?? [];
  $("one-font").hidden = !(fonts.length && fonts.length < 2);

  set("step", fmt.int(body.step));
  set("total", `/ ${fmt.int(body.total_steps)}`);
  set("stage", body.stage || "—");
  const done = body.total_steps ? body.step / body.total_steps : 0;
  $("fill").style.transform = `scaleX(${Math.min(done, 1)})`;

  set("loss", fmt.fixed(body.loss, 4));
  set("lr", fmt.exp(body.learning_rate));
  set("rate", body.rate == null ? "—" : `${body.rate.toFixed(1)} it/s`);
  set("elapsed", fmt.clock(body.elapsed_seconds));
  set("eta", body.state === "finished" ? "—" : fmt.clock(body.eta_seconds));

  const history = body.history ?? [];
  const last = history[history.length - 1];
  chart($("chart-synthetic"), history.map((h) => h.line_accuracy ?? 0), 0.5);
  set("char", last ? fmt.pct(last.char_accuracy) : "—");
  set("line", last ? fmt.pct(last.line_accuracy) : "—");
  set("eval-loss", last ? fmt.fixed(last.loss, 4) : "—");

  renderReal(body);

  set("device", body.hardware ? `${body.device} (${body.hardware})` : body.device || "—");
  set("precision", body.precision || "—");
  set("params", body.parameters ? `${body.parameters.toFixed(2)}M` : "—");
  set("workers", body.workers ? `${body.workers} / ${body.cores} cores` : "—");
  const fontCell = $("fonts");
  fontCell.textContent = fonts.length ? fonts.join(", ") : "—";
  fontCell.className = fonts.length && fonts.length < 2 ? "warn" : "";
}

function renderReal(body) {
  const real = body.real;
  $("real-empty").hidden = !!real;
  $("real-body").hidden = !real;
  if (!real) return;

  set("real-read", fmt.int(real.documents_read));
  set("real-total", `/ ${fmt.int(real.documents)}`);

  const trend = (body.real_history ?? []).map((h) =>
    h.documents ? h.documents_read / h.documents : 0
  );
  chart($("chart-real"), trend, 0);
  $("chart-real").className = "chart real";

  // One mark per document, so a set that is half-right looks different from one
  // where the same two always fail. Which two is the question worth having.
  const ticks = $("real-ticks");
  ticks.replaceChildren();
  for (const document_ of real.per_document ?? []) {
    const tick = document.createElement("span");
    tick.className = `tick ${document_.ok ? "ok" : "bad"}`;
    tick.title = `${document_.name}: ${document_.ok ? "exact" : "misread"}`;
    ticks.append(tick);
  }

  const list = $("real-confusions");
  list.replaceChildren();
  const confusions = real.confusions ?? [];
  if (!confusions.length) {
    const none = document.createElement("li");
    none.className = "none";
    none.textContent = "None — every character exact.";
    list.append(none);
    return;
  }
  for (const c of confusions) {
    const item = document.createElement("li");
    const want = document.createElement("span");
    want.textContent = c.want;
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "→";
    const got = document.createElement("span");
    got.textContent = c.got;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `×${c.count}`;
    item.append(want, arrow, got, count);
    list.append(item);
  }
}

async function poll() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error(String(response.status));
    render(await response.json());
  } catch {
    // The dashboard being unreachable says nothing about the run, so it must not
    // look like it does: the page keeps the last numbers and reports only its own
    // problem.
    $("run-dot").className = "dot bad";
    set("run-state", "Dashboard unreachable");
  }
}

poll();
setInterval(poll, POLL_MS);
