/* Upload, draw a box around the MRZ, read it.
 *
 * The one genuinely fiddly part is coordinates. The box is drawn in CSS pixels
 * over a scaled <img>, and the server crops the natural image, so every value
 * sent has to be converted by the ratio between the two. Getting that wrong
 * reads as a model that cannot see straight, so the conversion lives in exactly
 * one place: toNatural / toDisplay.
 */

const $ = (id) => document.getElementById(id);

const drop = $("drop");
const fileInput = $("file");
const stage = $("stage");
const preview = $("preview");
const canvas = $("canvas");
const sel = $("sel");
const found = $("found");
const readButton = $("read");

let file = null;
/** The selection, in CSS pixels relative to the image. */
let box = null;

/* ---------- coordinate conversion ---------- */

const scale = () => preview.naturalWidth / preview.clientWidth;

const toNatural = (b) => ({
  x: b.x * scale(),
  y: b.y * scale(),
  width: b.width * scale(),
  height: b.height * scale(),
});

const toDisplay = (b) => ({
  x: b.x / scale(),
  y: b.y / scale(),
  width: b.width / scale(),
  height: b.height / scale(),
});

/* ---------- selection box ---------- */

function paint() {
  if (!box) {
    sel.hidden = true;
    readButton.disabled = true;
    return;
  }
  sel.hidden = false;
  sel.style.left = `${box.x}px`;
  sel.style.top = `${box.y}px`;
  sel.style.width = `${box.width}px`;
  sel.style.height = `${box.height}px`;
  readButton.disabled = false;
}

/** An MRZ sits at the foot of the page. A first guess the user can correct
 *  beats an empty canvas and a puzzle.
 *
 *  Deliberately generous. The server finds the lines by their ink, so slack
 *  inside the box costs nothing, while a box that clips a character costs that
 *  character outright — the two errors are not symmetric, and the guess should
 *  not pretend they are. */
function guess() {
  box = {
    x: 0,
    y: preview.clientHeight * 0.66,
    width: preview.clientWidth,
    height: preview.clientHeight * 0.34,
  };
  paint();
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function pointIn(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: clamp(event.clientX - rect.left, 0, rect.width),
    y: clamp(event.clientY - rect.top, 0, rect.height),
  };
}

function fromCorners(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

let drag = null;

canvas.addEventListener("pointerdown", (event) => {
  if (stage.hidden) return;
  const at = pointIn(event);
  const corner = event.target.dataset.corner;

  if (corner) {
    // Resize: hold the opposite corner still and follow the pointer.
    const anchor = {
      x: corner.includes("w") ? box.x + box.width : box.x,
      y: corner.includes("n") ? box.y + box.height : box.y,
    };
    drag = { kind: "resize", anchor };
  } else if (event.target === sel) {
    drag = { kind: "move", grab: { x: at.x - box.x, y: at.y - box.y } };
  } else {
    drag = { kind: "draw", anchor: at };
    box = { x: at.x, y: at.y, width: 0, height: 0 };
  }
  canvas.setPointerCapture(event.pointerId);
  found.hidden = true;
  event.preventDefault();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const at = pointIn(event);

  if (drag.kind === "move") {
    box.x = clamp(at.x - drag.grab.x, 0, preview.clientWidth - box.width);
    box.y = clamp(at.y - drag.grab.y, 0, preview.clientHeight - box.height);
  } else {
    box = fromCorners(drag.anchor, at);
  }
  paint();
});

const endDrag = (event) => {
  if (!drag) return;
  drag = null;
  canvas.releasePointerCapture?.(event.pointerId);
  // A stray click is not a selection; the guess is more useful than a dot.
  if (box && (box.width < 12 || box.height < 12)) guess();
};

canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

/* ---------- upload ---------- */

function load(chosen) {
  if (!chosen || !chosen.type.startsWith("image/")) {
    return toast("That file is not an image.");
  }
  file = chosen;
  const url = URL.createObjectURL(file);
  preview.onload = () => {
    URL.revokeObjectURL(url);
    drop.hidden = true;
    stage.hidden = false;
    $("reset").hidden = false;
    found.hidden = true;
    guess();
  };
  preview.src = url;
}

drop.addEventListener("click", () => fileInput.click());
$("browse").addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener("change", () => load(fileInput.files[0]));

["dragenter", "dragover"].forEach((name) =>
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("over");
  })
);
["dragleave", "drop"].forEach((name) =>
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.remove("over");
  })
);
drop.addEventListener("drop", (event) => load(event.dataTransfer.files[0]));

$("reset").addEventListener("click", () => {
  file = null;
  box = null;
  preview.removeAttribute("src");
  fileInput.value = "";
  stage.hidden = true;
  drop.hidden = false;
  $("reset").hidden = true;
  $("result").hidden = true;
  $("empty").hidden = false;
  $("verdict").hidden = true;
});

// The box is in CSS pixels, so a resized window invalidates it.
window.addEventListener("resize", () => {
  if (box) paint();
});

/* ---------- read ---------- */

readButton.addEventListener("click", async () => {
  if (!file || !box) return;
  const natural = toNatural(box);

  const form = new FormData();
  form.append("image", file);
  for (const key of ["x", "y", "width", "height"]) form.append(key, natural[key]);

  stage.classList.add("busy");
  readButton.textContent = "Reading";
  try {
    const response = await fetch("/api/read", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The read failed.");
    render(body);
  } catch (error) {
    toast(error.message);
  } finally {
    stage.classList.remove("busy");
    readButton.textContent = "Read MRZ";
  }
});

/* ---------- render ---------- */

function render(body) {
  $("empty").hidden = true;
  $("result").hidden = false;

  const verdict = $("verdict");
  verdict.hidden = false;
  verdict.textContent = body.valid ? "Valid" : "Failed checks";
  verdict.className = `chip ${body.valid ? "pass" : "fail"}`;

  // Say this before the fields, not after: if the box cut a line off, the
  // reading below is a guess at characters the model was never shown, and no
  // amount of staring at the fields will reveal that.
  const warning = $("clipped");
  warning.hidden = !body.clipped;

  const fields = $("fields");
  fields.replaceChildren();
  for (const field of body.fields) {
    const row = document.createElement("div");
    row.className = `field${field.status === "failed" ? " is-failed" : ""}`;

    const label = document.createElement("dt");
    label.textContent = field.label;

    const value = document.createElement("dd");
    value.textContent = field.value;

    const tag = document.createElement("span");
    tag.className = `tag ${field.status}`;
    tag.textContent = field.status;
    tag.title =
      field.guard === "none"
        ? "No check digit covers this field. Read it against the page yourself."
        : field.guard === "checksum"
        ? "Covered by an ICAO check digit."
        : "Checked against the list of valid codes.";

    row.append(label, value, tag);
    fields.append(row);
  }

  $("mrz").textContent = `${body.mrz.line1}\n${body.mrz.line2}`;

  const issues = $("issues");
  issues.replaceChildren();
  $("issues-block").hidden = body.issues.length === 0;
  for (const issue of body.issues) {
    const item = document.createElement("li");
    const field = document.createElement("code");
    field.textContent = issue.field;
    item.append(field, document.createTextNode(` ${issue.message}`));
    issues.append(item);
  }

  drawFound(body.lines, body.valid, body.skew);
}

/** Outline the crops the server actually read.
 *  A wrong reading over a visibly wrong crop explains itself; the same reading
 *  without it looks like a model that cannot read.
 *
 *  The boxes arrive in the frame the server levelled, so they are turned back by
 *  the tilt it removed — otherwise a correct reading of a tilted page would be
 *  drawn beside the text it actually read, and look like the bug it is not.
 *  Scale-invariant: an angle is an angle at any zoom, and the pivot divides
 *  through like every other coordinate. */
function drawFound(lines, valid, skew) {
  found.replaceChildren();
  found.hidden = false;
  const pivot = skew && skew.deg ? { x: skew.x / scale(), y: skew.y / scale() } : null;

  lines.forEach((line, index) => {
    const shown = toDisplay(line);
    const element = document.createElement("div");
    element.className = `found-line${valid ? "" : " bad"}`;
    element.style.left = `${shown.x}px`;
    element.style.top = `${shown.y}px`;
    element.style.width = `${shown.width}px`;
    element.style.height = `${shown.height}px`;
    if (pivot) {
      // CSS turns clockwise for a positive angle, which is the way back: the
      // server turned the page the other way to level it.
      element.style.transformOrigin = `${pivot.x - shown.x}px ${pivot.y - shown.y}px`;
      element.style.transform = `rotate(${skew.deg}deg)`;
    }

    const number = document.createElement("span");
    number.textContent = index + 1;
    element.append(number);
    found.append(element);
  });
}

/* ---------- chrome ---------- */

let toastTimer = null;
function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (element.hidden = true), 4500);
}

(async function health() {
  const dot = $("status-dot");
  const text = $("status-text");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    await response.json();
    dot.className = "dot ok";
    text.textContent = "Model ready";
  } catch {
    dot.className = "dot bad";
    text.textContent = "Model unavailable";
  }
})();
