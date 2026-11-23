/*! Copyright 2026 The IREE Authors. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

const COLORS = {
  output: "#69d9da",
  input: "#f8aa74",
  active: "#aaa0e8",
  memory: "#afcf87",
  errors: "#f58c97",
};
const RATE_WINDOW_NS = 5000000000n;
const ACTIVITY_BUCKET_NS = 1000000000n;
let windowMs = 30000;
const decimalFormat = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 1,
});
const integerFormat = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});
const encoder = new TextEncoder();

// Fixed history, independent of server uptime, counter magnitude, and traffic.
class Ring {
  constructor(color) {
    this.color = color;
    this.times = new Float64Array(512);
    this.values = new Float64Array(512);
    this.clear();
  }
  clear() {
    this.count = 0;
    this.next = 0;
    this.peak = NaN;
  }
  push(time, value) {
    this.times[this.next] = time;
    this.values[this.next] = value;
    this.next = (this.next + 1) % this.times.length;
    this.count = Math.min(this.count + 1, this.times.length);
    if (Number.isFinite(value)) this.peak = Math.max(this.peak || 0, value);
  }
  each(callback) {
    for (let i = 0; i < this.count; ++i) {
      const index =
        (this.next - this.count + i + this.times.length) % this.times.length;
      callback(this.times[index], this.values[index]);
    }
  }
}

// Retain exact counters, not an unweighted mean of interval rates. The oldest
// sample brackets the five-second boundary; warm-up uses only observed time.
// Each sample enters and leaves this fixed window once. A reset or an interval
// longer than the window starts a new baseline instead of inventing activity.
class CounterRate {
  constructor() {
    this.times = new BigUint64Array(64);
    this.values = new BigUint64Array(64);
    this.first = 0;
    this.count = 0;
    this.peak = NaN;
    this.value = NaN;
  }
  reset() {
    this.count = 0;
    this.value = NaN;
  }
  push(time, raw, unchangedUntil = 0n) {
    if (raw === undefined || raw === null) {
      this.reset();
      return NaN;
    }
    const value = BigInt(raw);
    if (this.count) {
      const last = (this.first + this.count - 1) % this.times.length;
      // A matching source revision proves the old value persisted through a
      // sampled boundary. Bridge quiet periods, never an unobserved gap or a
      // reconnect, so the first burst after idleness remains measurable.
      if (
        time - this.times[last] > RATE_WINDOW_NS &&
        unchangedUntil >= time - RATE_WINDOW_NS &&
        unchangedUntil > this.times[last] &&
        value >= this.values[last]
      ) {
        this.first = last;
        this.count = 1;
        this.times[last] = time - RATE_WINDOW_NS;
      }
      if (
        time <= this.times[last] ||
        time - this.times[last] > RATE_WINDOW_NS ||
        value < this.values[last]
      )
        this.reset();
    }
    if (this.count === this.times.length) {
      this.first = (this.first + 1) % this.times.length;
      --this.count;
    }
    const index = (this.first + this.count++) % this.times.length;
    this.times[index] = time;
    this.values[index] = value;
    while (this.count > 1) {
      const next = (this.first + 1) % this.times.length;
      if (this.times[next] > time - RATE_WINDOW_NS) break;
      this.first = next;
      --this.count;
    }
    const elapsed = time - this.times[this.first];
    this.value =
      elapsed > 0n
        ? Number(value - this.values[this.first]) / (Number(elapsed) / 1e9)
        : NaN;
    // A short warm-up interval is not a five-second throughput peak. Baseline
    // resets preserve observed peaks; replacing the source replaces this object.
    if (elapsed >= RATE_WINDOW_NS)
      this.peak = Math.max(this.peak || 0, this.value);
    return this.value;
  }
}

// Nonoverlapping count intervals close on real observations, never interpolated
// event times. The actual interval can exceed one second by the sampling delay.
// Retaining the completed bucket avoids replacing a burst with partial zeros.
class CounterBucket {
  constructor() {
    this.peak = NaN;
    this.reset();
  }
  reset() {
    this.lastTime = undefined;
    this.value = NaN;
    this.duration = 0n;
  }
  push(time, raw, unchangedUntil = 0n) {
    if (raw === undefined || raw === null) {
      this.reset();
      return true;
    }
    const value = BigInt(raw);
    if (this.lastTime !== undefined) {
      if (time <= this.lastTime || value < this.lastValue) this.reset();
      else if (time - this.lastTime > RATE_WINDOW_NS) {
        if (unchangedUntil >= time - ACTIVITY_BUCKET_NS) {
          this.start = time - ACTIVITY_BUCKET_NS;
          this.baseline = this.lastValue;
        } else this.reset();
      }
    }
    const first = this.lastTime === undefined;
    this.lastTime = time;
    this.lastValue = value;
    if (first) {
      this.start = time;
      this.baseline = value;
      return true;
    }
    if (time - this.start < ACTIVITY_BUCKET_NS) return false;
    this.value = Number(value - this.baseline);
    this.duration = time - this.start;
    this.start = time;
    this.baseline = value;
    this.peak = Math.max(this.peak || 0, this.value);
    return true;
  }
}

function lifetimeRate(time, since, raw) {
  if (
    since === undefined ||
    since === null ||
    raw === undefined ||
    raw === null
  )
    return NaN;
  const elapsed = time - BigInt(since);
  return elapsed > 0n ? Number(BigInt(raw)) / (Number(elapsed) / 1e9) : NaN;
}

const $ = (id) => document.getElementById(id);
const format = (value) =>
  Number.isFinite(value)
    ? (value < 10 ? decimalFormat : integerFormat).format(value)
    : "—";
const formatAverage = (value) =>
  value > 0 && value < 0.1 ? value.toPrecision(2) : format(value);
const integer = (value) => integerFormat.format(BigInt(value));
const scaled = (value, base, units) => {
  if (!Number.isFinite(value)) return "—";
  let unit = 0;
  while (value >= base && unit < units.length - 1) {
    value /= base;
    ++unit;
  }
  return `${format(value)} ${units[unit]}`;
};
const bytes = (value) => scaled(value, 1024, ["B", "KiB", "MiB", "GiB", "TiB"]);
const history = {
  output: new Ring(COLORS.output),
  input: new Ring(COLORS.input),
  request: new Ring(COLORS.input),
  arrivals: new Ring(COLORS.active),
  finishes: new Ring(COLORS.input),
  kv: new Ring(COLORS.memory),
  errors: new Ring(COLORS.errors),
};
const plots = Array.from(document.querySelectorAll("canvas[data-series]"), (canvas) => ({
  canvas,
  series: canvas.dataset.series.split(" ").map((key) => history[key]),
  grid: canvas.classList.contains("chart"),
}));
const models = new Map();
const devices = new Map();
const pools = new Map();
const pendingPlots = new Set();
const serverRates = Array.from(document.querySelectorAll("[data-rate]"), (card) => {
  const key = card.querySelector("canvas").dataset.series;
  const summary = $("rate-summary").content.cloneNode(true);
  const [peak, average] = summary.querySelectorAll("b");
  peak.id = key + "-peak";
  average.id = key + "-average";
  const value = card.querySelector("strong");
  value.parentElement.append(summary);
  return {
    counter: card.dataset.rate,
    ring: history[key],
    rate: new CounterRate(),
    value, peak, average,
    total: card.querySelector("small span"),
  };
});
const serverTotals = document.querySelectorAll("[data-total]");
const serverBuckets = {
  arrivals: new CounterBucket(),
  finishes: new CounterBucket(),
};
let activePeak = 0;
let socket;
let paused = false;
let scrolling = true;
let pausedTime = 0;
let retry;
let retryDelay = 1000;
let serverIdentity;
let serverSample;
let serverRevision;
let latestTime = 0;
let historyTime = 0;
let receivedAt = 0;
let sequence = 0n;
let receivedBytes = 0;
let receivedFrames = 0;
let framePending = false;

function chartTime(now = performance.now()) {
  if (paused) return pausedTime;
  return scrolling && receivedAt
    ? latestTime + Math.max(0, now - receivedAt)
    : historyTime;
}

function draw(canvas, series, now, grid = false) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = grid ? 35 : 0;
  const top = grid ? 10 : 3;
  const bottom = height - 3;
  const extend = socket?.readyState === WebSocket.OPEN && !paused;
  let maximum = 1;
  for (const ring of series) {
    let last = NaN;
    ring.each((time, value) => {
      if (time <= now) last = value;
      if (time >= now - windowMs && time <= now && Number.isFinite(value))
        maximum = Math.max(maximum, value);
    });
    if (extend && Number.isFinite(last)) maximum = Math.max(maximum, last);
  }
  maximum *= 1.15;
  if (grid) {
    context.font = "9px ui-monospace, monospace";
    context.fillStyle = "#6f7c8d";
    context.strokeStyle = "#292f38";
    context.lineWidth = 0.7;
    for (let i = 0; i <= 4; ++i) {
      const y = top + ((bottom - top) * i) / 4;
      context.fillText(format(maximum * (1 - i / 4)), 0, y + 3);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width, y);
      context.stroke();
    }
  }
  const x = (time) => left + ((time - now + windowMs) / windowMs) * (width - left);
  const y = (value) => bottom - (value / maximum) * (bottom - top);
  for (const ring of series) {
    const color = ring.color;
    const gradient = context.createLinearGradient(0, top, 0, bottom);
    gradient.addColorStop(0, `${color}35`);
    gradient.addColorStop(1, `${color}00`);
    context.strokeStyle = color;
    context.lineWidth = grid ? 1.7 : 1.4;
    let line, firstX, lastTime, lastValue;
    function paintSegment() {
      if (!line) return;
      const area = new Path2D(line);
      const lastX = x(lastTime);
      area.lineTo(lastX, bottom);
      area.lineTo(firstX, bottom);
      area.closePath();
      context.fillStyle = gradient;
      context.fill(area);
      context.stroke(line);
      context.beginPath();
      context.arc(lastX, y(lastValue), grid ? 2.8 : 2, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      line = undefined;
    }
    // A missing observation or two seconds of silence ends the current path.
    ring.each((time, value) => {
      if (time > now) return;
      if (!Number.isFinite(value) || time - lastTime > 2000) paintSegment();
      if (time >= now - windowMs && Number.isFinite(value)) {
        if (line) line.lineTo(x(time), y(value));
        else {
          line = new Path2D();
          firstX = x(time);
          line.moveTo(firstX, y(value));
        }
      }
      lastTime = time;
      lastValue = value;
    });
    paintSegment();
    // Local time moves the viewport, not the observations. A dotted tail shows
    // only the last received value; it never enters history, rates, or peaks.
    if (Number.isFinite(lastValue) && extend) {
      context.beginPath();
      context.moveTo(Math.max(left, x(lastTime)), y(lastValue));
      context.lineTo(width, y(lastValue));
      context.setLineDash([2, 4]);
      context.lineWidth = 1;
      context.strokeStyle = color;
      context.globalAlpha = 0.5;
      context.stroke();
      context.globalAlpha = 1;
      context.setLineDash([]);
    }
  }
}

function redraw(all = false) {
  if (all) {
    for (const model of models.values()) pendingPlots.add(model);
    for (const pool of pools.values()) pendingPlots.add(pool);
  }
  if (framePending || document.hidden) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    if (document.hidden) return;
    const now = chartTime();
    for (const { canvas, series, grid } of plots) draw(canvas, series, now, grid);
    for (const plot of pendingPlots) draw(plot.canvas, [plot.ring], now);
    pendingPlots.clear();
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Clone trusted page markup once per inventory card. Observation strings only
// enter text nodes. The bound fields survive every subsequent observation.
function mountCard(kind, inventory) {
  const card = $(kind + "-card").content.firstElementChild.cloneNode(true);
  const grid = $(kind + "-grid");
  if (!inventory.size) grid.replaceChildren();
  grid.append(card);
  const state = { card };
  for (const node of card.querySelectorAll("[data-field]")) {
    state[node.dataset.field] = node;
  }
  return state;
}

function updateTopicFields(topics, state) {
  const fields = new Map();
  for (const topic of topics)
    for (const field of topic.fields) {
      const key = `${topic.topic}.${field.name}`;
      fields.set(key, field);
      if (!state.cells.has(key)) {
        const row = element("tr");
        const cell = element("td");
        row.append(element("td", "", key.replaceAll("_", " ")), cell);
        state.table.append(row);
        state.cells.set(key, cell);
      }
      const cell = state.cells.get(key);
      cell.textContent =
        field.unit === "bytes"
          ? bytes(Number(field.value))
          : field.unit === "nanoseconds"
            ? scaled(Number(field.value), 1000, ["ns", "µs", "ms", "s"])
            : `${integer(field.value)} ${field.unit}`;
      cell.title = `${field.value} ${field.unit}`;
    }
  for (const [key, cell] of state.cells) {
    if (!fields.has(key)) {
      cell.parentElement.remove();
      state.cells.delete(key);
    }
  }
  return fields;
}

function resourceName(source) {
  return source.key + (source.key_truncated ? "…" : "");
}

function mountResource(kind, inventory, source) {
  const state = mountCard(kind, inventory);
  state.source = source;
  state.card.id = `${kind}-${source.index}`;
  state.name.textContent = resourceName(source);
  return state;
}

function place(link, kind, index, inventory, unassigned) {
  const resource = inventory.get(index);
  link.textContent = resource
    ? resourceName(resource.source)
    : index == null ? unassigned : `${kind} ${index}`;
  if (resource) link.href = `#${kind}-${index}`;
  else link.removeAttribute("href");
}

function summarizeDevices() {
  const activity = new Map(
    [...devices.keys()].map((index) => [
      index,
      { models: 0, active: 0, queued: 0, held: 0 },
    ]),
  );
  for (const state of models.values()) {
    place(state.placement, "device", state.model.device, devices, "unassigned");
    place(state.poolPlacement, "pool", state.model.pool, pools, "unassigned pool");
    const totals = activity.get(state.model.device);
    if (!totals) continue;
    ++totals.models;
    totals.active += state.model.active;
    totals.queued += state.model.queued;
    totals.held += state.model.held;
  }
  for (const [index, totals] of activity) {
    devices.get(index).activity.textContent =
      `${totals.models} models · ${totals.active} active · ${totals.queued} queued · ${totals.held} held`;
  }
}

// Immutable inventory edges are rebuilt only when a new resource arrives.
// This also handles a reader receiving the pool before a referenced device.
function updateResourceLinks() {
  for (const device of devices.values()) device.links.replaceChildren();
  for (const pool of pools.values()) {
    pool.links.replaceChildren();
    for (const index of pool.source.devices) {
      const device = devices.get(index);
      const link = element(
        "a", "device-link",
        device ? resourceName(device.source) : `device ${index}`,
      );
      link.href = `#device-${index}`;
      pool.links.append(link);
      if (device) {
        const owner = element("a", "device-link", resourceName(pool.source));
        owner.href = `#pool-${pool.source.index}`;
        device.links.append(owner);
      }
    }
  }
}

function slabState(slab) {
  if (!slab) return "returned";
  if (slab.state !== "ready") return slab.state;
  if (!slab.leased) return "reusable";
  return slab.leased === slab.pages ? "leased" : "mixed";
}

// Only an open inspector materializes the latest observation's options and bits.
function createSlabMap(details) {
  const summary = details.querySelector("summary");
  const select = details.querySelector("select");
  const pages = details.querySelector(".tiles");
  const caption = details.querySelector(".model-facts");
  let source, selected, choiceKey, pageKey;

  function showPages() {
    const slab = source.slabs[select.selectedIndex];
    const key = (source.first_slab + select.selectedIndex) + "/" + JSON.stringify(slab);
    if (key === pageKey) return;
    pageKey = key;
    pages.replaceChildren();
    if (!slab) {
      caption.textContent = "Slab has left the pool";
      return;
    }
    caption.textContent = `${integer(slab.leased)}/${integer(slab.pages)} pages leased · ${bytes(slab.page_bytes)} per page`;
    if (slabState(slab) !== "mixed") return;
    const count = Math.min(64, slab.pages - slab.offset);
    const free = BigInt("0x" + slab.free);
    for (let i = 0; i < count; ++i) {
      const cell = element("i");
      cell.dataset.state = free & (1n << BigInt(i)) ? "reusable" : "leased";
      cell.title = `Page ${slab.offset + i} · ${cell.dataset.state}`;
      pages.append(cell);
    }
    caption.textContent += ` · showing #${integer(slab.offset)}–${integer(slab.offset + count - 1)}`;
  }

  function inspect() {
    if (!details.open || !source.slabs.length) return;
    const first = source.first_slab;
    const states = source.slabs.map(slabState);
    const key = first + "/" + states.join();
    if (key !== choiceKey) {
      choiceKey = key;
      select.replaceChildren(...states.map((state, i) => new Option(`#${first + i} · ${state}`)));
    }
    let index = selected - first;
    if (!(index >= 0 && index < states.length)) {
      index = states.indexOf("mixed");
      if (index < 0) index = Math.max(0, source.slabs.findLastIndex(Boolean));
    }
    select.selectedIndex = index;
    showPages();
  }

  details.ontoggle = inspect;
  select.onchange = () => {
    selected = source.first_slab + select.selectedIndex;
    showPages();
  };
  return next => {
    source = next;
    details.hidden = !next.slabs.length;
    summary.textContent = `Recent physical slabs · #${next.first_slab}–${next.first_slab + next.slabs.length - 1}`;
    inspect();
  };
}

function createPool(source) {
  const pool = mountResource("pool", pools, source);
  pool.backing.textContent = source.backing;
  pool.cells = new Map();
  pool.ring = new Ring(COLORS.memory);
  pool.parts = ["held", "reusable", "uncommitted"].map((state) => {
    const segment = element("i");
    const label = element("span");
    segment.dataset.state = label.dataset.state = state;
    pool.capacity.append(segment);
    pool.legend.append(label);
    return { segment, label, state };
  });
  if (source.slabs) pool.updateSlabs = createSlabMap(pool.slabs);
  else pool.slabs.remove();
  return pool;
}

function updateDevice(device) {
  if (!devices.has(device.index)) {
    const state = mountResource("device", devices, device);
    state.identity.textContent = `${device.name} · ${device.driver} / ${device.backend}`;
    devices.set(device.index, state);
    updateResourceLinks();
  }
  $("device-count").textContent = devices.size;
  summarizeDevices();
}

function updatePool(source, time) {
  let pool = pools.get(source.index);
  if (!pool) {
    pool = createPool(source);
    pools.set(source.index, pool);
    updateResourceLinks();
    $("pool-count").textContent = pools.size;
    summarizeDevices();
  }
  pool.updateSlabs?.(source);
  const fields = updateTopicFields(source.topics, pool);
  const value = (topic, name) => Number(fields.get(`${topic}.${name}`)?.value);
  const elastic = source.backing === "elastic";
  const owner = elastic ? "elastic" : "storage";
  const committed = value(owner, "committed_bytes");
  const limit = value(owner, "budget_limit");
  const reusable = value(owner, elastic ? "cached_bytes" : "free_bytes");
  pool.committed.textContent = bytes(committed);
  pool.budget.textContent = limit
    ? `committed of ${bytes(limit)} budget`
    : "committed";
  // Ready backing is a subset of commitment. Workspace has its own view of
  // these bytes and its counters never enter this partition.
  const scale = Math.max(limit, committed);
  const amounts = [committed - reusable, reusable, Math.max(0, limit - committed)];
  for (const [i, {segment, label, state}] of pool.parts.entries()) {
    segment.style.width = (scale ? 100 * amounts[i] / scale : 0) + "%";
    label.hidden = i === 2 && !limit;
    label.textContent = `${bytes(amounts[i])} ${state}`;
  }
  pool.ring.push(time, reusable);
  pendingPlots.add(pool);
  pool.peak.textContent = `peak ${bytes(pool.ring.peak)}`;
  const count = name => BigInt(fields.get("workspace." + name + "_count").value);
  const reserves = count("reserve");
  pool.initialReserves ??= reserves;
  const waits = count("wait");
  pool.workspace.textContent = `Workspace · ${bytes(value("workspace", "committed_bytes"))} retained · ${integer(count("reuse"))} reused · ${integer(count("fresh"))} fresh`
    + (waits ? ` · ${integer(waits)} waited` : "")
    + ` · +${integer(reserves - pool.initialReserves)} reservations since opening`;
}

const batchWork = {
  unit: "Batches", color: COLORS.output, total: (model) => model.counters?.batches, facts: [],
};
const modelWork = new Map([
  ["text", {
    unit: "Tokens", color: COLORS.output, total: (model) => model.counters?.output_tokens,
    facts: [["text.session_capacity", "rows"], ["text.session_cache_endpoints", "cached sessions"],
      ["text.prefix_cache_endpoints", "cached prefixes"]],
  }],
  ["image", {
    unit: "Images", color: COLORS.input, total: (_, fields) => fields.get("image.generated_images")?.value,
    facts: [["image.generated_pixels", "pixels generated"]],
  }],
]);

function createModel(model) {
  const state = mountCard("model", models);
  state.name.textContent = model.name + (model.name_truncated ? "…" : "");
  state.kind.prepend(model.kind);
  state.canvas.ariaLabel = `${model.name} completed work rate`;
  state.format = modelWork.get(model.kind) ?? batchWork;
  function note(item, kind, title) {
    const text = element("span", kind, kind === "peak" ? "peak " : "avg ");
    const number = element("span", "", "—");
    text.title = title;
    text.append(number);
    item.append(text);
    return number;
  }
  for (const [key, label, kind] of [
    ["active", "Active now", "gauge"],
    ["queued", "Queued + held", "gauge"],
    ["completed", "Completed"],
    ["requests", "Requests/s · 5s", "rate"],
    ["work", `${state.format.unit}/s · 5s`, "rate"],
    ["total", `${state.format.unit} total`],
  ]) {
    const item = element("div");
    const field = state[key] = { value: element("strong", "", "—") };
    item.append(element("label", "", label), field.value);
    if (kind) field.peak = note(item, "peak",
      "Observed peak in this page session; rates require a full five-second window.");
    if (kind === "rate") {
      field.average = note(item, "average",
        "Lifetime average, including idle time; independent of this page session.");
      field.rate = new CounterRate();
    }
    state.metrics.append(item);
  }
  state.requests.total = state.completed.value;
  state.work.total = state.total.value;
  state.id = model.incarnation;
  state.cells = new Map();
  state.ring = new Ring(state.format.color);
  return state;
}

function updateGauge(field, value) {
  field.maximum = Math.max(field.maximum || 0, value);
  field.value.textContent = format(value);
  field.peak.textContent = format(field.maximum);
}

function updateRate(field, total, timeNs, since, unchangedUntil) {
  const rate = field.rate.push(timeNs, total, unchangedUntil);
  field.value.textContent = format(rate);
  field.peak.textContent = format(field.rate.peak);
  field.average.textContent = formatAverage(lifetimeRate(timeNs, since, total));
  field.total.textContent = total == null ? "—" : integer(total);
  return rate;
}

// Only an exact revision match authorizes extending the last observed value.
// Skipped revisions remain a gap even when the final counters happen to match.
function unchangedBoundary(previous, version, timeNs) {
  if (previous === version.revision) return timeNs;
  return previous === version.previous_revision
    ? BigInt(version.previous_time_ns)
    : 0n;
}

function updateModel(model, time, timeNs, version, recordHistory) {
  let state = models.get(model.slot);
  if (state && state.id !== model.incarnation) {
    pendingPlots.delete(state);
    state.card.remove();
    models.delete(model.slot);
    state = null;
  }
  if (!state) {
    state = createModel(model);
    models.set(model.slot, state);
  }
  const fields = updateTopicFields(model.topics, state);
  const value = (key) => Number(fields.get(key)?.value);
  const workTotal = state.format.total(model, fields);
  const unchangedUntil = unchangedBoundary(state.revision, version, timeNs);
  const workRate = updateRate(state.work, workTotal, timeNs, model.since_ns, unchangedUntil);
  updateRate(state.requests, model.counters?.completed, timeNs, model.since_ns, unchangedUntil);
  state.revision = version.revision;
  if (recordHistory) {
    state.ring.push(time, workRate);
    pendingPlots.add(state);
  }
  state.model = model;
  state.fields = fields;
  state.phase.textContent = model.phase;
  updateGauge(state.active, model.active);
  updateGauge(state.queued, model.queued + model.held);
  const capacity = value("text.pool_capacity");
  state.pool.hidden = !(capacity > 0);
  if (capacity > 0) {
    const resident = value("text.pool_resident");
    const reserved = value("text.pool_reserved");
    state.poolValue.textContent = `${format(resident)} resident + ${format(reserved)} reserved / ${format(capacity)}`;
    state.resident.style.width = `${(100 * resident) / capacity}%`;
    state.reserved.style.width = `${(100 * reserved) / capacity}%`;
  }
  state.cache.textContent = state.format.facts
    .filter(([key]) => fields.has(key))
    .map(([key, label]) => `${integer(fields.get(key).value)} ${label}`)
    .join(" · ");
  state.outcomes.textContent = model.counters
    ? `${integer(model.counters.failed)} failed · ${integer(model.counters.rejected)} rejected · ${integer(model.counters.cancelled)} cancelled`
    : "";
}

function summarize(time, recordHistory) {
  summarizeDevices();
  let active = 0,
    queued = 0,
    held = 0,
    capacity = 0,
    resident = 0,
    reserved = 0,
    cachedSessions = 0,
    cachedPrefixes = 0,
    hasCache = false;
  const mix = new Map();
  for (const state of models.values()) {
    const model = state.model;
    active += model.active;
    queued += model.queued;
    held += model.held;
    const group = mix.get(model.kind) || {
      active: 0,
      queued: 0,
      rate: 0,
      available: false,
      completed: 0n,
      hasCounters: false,
    };
    group.active += model.active;
    group.queued += model.queued + model.held;
    if (Number.isFinite(state.requests.rate.value)) {
      group.rate += state.requests.rate.value;
      group.available = true;
    }
    if (model.counters) {
      group.completed += BigInt(model.counters.completed);
      group.hasCounters = true;
    }
    mix.set(model.kind, group);
    const value = (key) => Number(state.fields.get(`text.${key}`)?.value ?? 0);
    capacity += value("pool_capacity");
    resident += value("pool_resident");
    reserved += value("pool_reserved");
    cachedSessions += value("session_cache_endpoints");
    cachedPrefixes += value("prefix_cache_endpoints");
    hasCache ||= state.fields.has("text.session_cache_endpoints");
  }
  $("active").textContent = format(active);
  activePeak = Math.max(activePeak, active);
  $("active-peak").textContent = format(activePeak);
  $("queued").textContent = format(queued);
  $("held").textContent = format(held);
  const occupancy = capacity ? (100 * resident) / capacity : NaN;
  $("kv").textContent = format(occupancy);
  $("kv-resident").textContent = capacity ? format(resident) : "—";
  $("kv-capacity").textContent = capacity ? format(capacity) : "—";
  $("kv-reserved").textContent = capacity ? format(reserved) : "—";
  $("cached-sessions").textContent = hasCache ? format(cachedSessions) : "—";
  $("cached-prefixes").textContent = hasCache ? format(cachedPrefixes) : "—";
  $("model-count").textContent = models.size;
  if (recordHistory) {
    history.kv.push(time, occupancy);
  }
  $("kv-peak").textContent = format(history.kv.peak);
  const rows = [];
  let totalRate = 0;
  for (const item of mix.values()) totalRate += item.rate;
  for (const [kind, values] of mix) {
    const row = element("div", "mix-row");
    const labels = element("div", "labels");
    labels.append(
      element("span", "", kind),
      element(
        "span",
        "",
        `${format(values.available ? values.rate : NaN)} req/s`,
      ),
    );
    const bar = element("div", "bar");
    const run = element("span");
    run.style.width = `${totalRate ? (100 * values.rate) / totalRate : 0}%`;
    bar.append(run);
    row.append(
      labels,
      bar,
      element(
        "small",
        "",
        `${values.hasCounters ? integer(values.completed) : "—"} completed · ${values.active} active · ${values.queued} queued`,
      ),
    );
    rows.push(row);
  }
  $("mix").replaceChildren(
    ...(rows.length ? rows : [element("p", "empty", "No registered models.")]),
  );
  if (!models.size)
    $("model-grid").replaceChildren(
      element("div", "panel empty", "No registered models."),
    );
}

function updateServer(page, time, timeNs, version, recordHistory) {
  const unchangedUntil = unchangedBoundary(serverRevision, version, timeNs);
  serverRevision = version.revision;
  serverSample = page;
  for (const field of serverRates) {
    const value = updateRate(field, page.counters?.[field.counter], timeNs, page.since_ns, unchangedUntil);
    if (recordHistory) field.ring.push(time, value);
  }
  const finishes = page.counters
    ? ["completed", "rejected", "cancelled", "failed"].reduce(
        (total, key) => total + BigInt(page.counters[key]),
        0n,
      )
    : undefined;
  for (const [key, total] of [
    ["arrivals", page.counters?.requests],
    ["finishes", finishes],
  ]) {
    const bucket = serverBuckets[key];
    if (bucket.push(timeNs, total, unchangedUntil) && recordHistory)
      history[key].push(time, bucket.value);
  }
  $("activity").textContent = format(serverBuckets.arrivals.value);
  $("activity-peak").textContent = format(serverBuckets.arrivals.peak);
  $("activity-span").textContent = serverBuckets.arrivals.duration
    ? `${(Number(serverBuckets.arrivals.duration) / 1e9).toFixed(2)}s bucket`
    : "~1s bucket";
  if (recordHistory) history.errors.push(time, Number(page.counters?.failed));
  for (const total of serverTotals)
    total.textContent = page.counters ? integer(page.counters[total.dataset.total]) : "—";
}

function apply(frame) {
  if (frame.version !== 1 || BigInt(frame.sequence) !== sequence + 1n)
    throw Error("Unsupported or discontinuous observation stream");
  sequence = BigInt(frame.sequence);
  if (serverIdentity !== frame.server) {
    serverIdentity = frame.server;
    serverRevision = undefined;
    serverSample = undefined;
    for (const field of serverRates) field.rate = new CounterRate();
    for (const key of Object.keys(serverBuckets))
      serverBuckets[key] = new CounterBucket();
    activePeak = 0;
    pendingPlots.clear();
    models.clear();
    devices.clear();
    pools.clear();
    $("model-grid").replaceChildren();
    $("device-grid").replaceChildren();
    $("pool-grid").replaceChildren();
    $("device-count").textContent = "0";
    $("pool-count").textContent = "0";
    for (const ring of Object.values(history)) ring.clear();
    latestTime = 0;
    historyTime = 0;
  }
  const timeNs = BigInt(frame.time_ns);
  const time = Number(timeNs / 1000000n);
  latestTime = Math.max(latestTime, time);
  receivedAt = performance.now();
  const page = frame.page;
  // Settled heartbeats refresh facts without evicting the last activity window.
  // Only real samples populate history; the local scroll clock adds no samples.
  const recordHistory =
    page.type !== "heartbeat" ||
    serverRates.some(({ rate }) => rate.count && rate.value !== 0) ||
    Object.values(serverBuckets).some(
      (bucket) => Number.isFinite(bucket.value) && bucket.value !== 0,
    ) ||
    [...models.values()].some(
      (model) =>
        (model.work.rate.count && model.work.rate.value !== 0) ||
        (model.requests.rate.count && model.requests.rate.value !== 0),
    );
  if (recordHistory) historyTime = latestTime;
  if (page.type === "server") {
    updateServer(page, time, timeNs, frame, recordHistory);
    if (!page.device_count)
      $("device-grid").textContent = "No application-owned devices.";
    if (!page.pool_count)
      $("pool-grid").textContent = "No application-owned pools.";
  } else if (page.type === "device") {
    updateDevice(page);
  } else if (page.type === "pool") {
    updatePool(page, time);
  } else if (page.type === "models") {
    // A live page establishes absence only for its explicitly scanned slots.
    for (let slot = page.first; slot < page.next; ++slot) {
      const model = models.get(slot);
      if (model && slot !== page.model?.slot) {
        model.card.remove();
        pendingPlots.delete(model);
        models.delete(slot);
      }
    }
    if (page.model) updateModel(page.model, time, timeNs, frame, recordHistory);
    if (page.done) summarize(time, recordHistory);
  } else if (page.type === "heartbeat") {
    // All pages have been applied and the server has observed no content
    // change. This advances known idleness, not an estimate from silence.
    updateServer(
      serverSample,
      time,
      timeNs,
      { revision: serverRevision },
      recordHistory,
    );
    for (const state of models.values())
      updateModel(
        state.model,
        time,
        timeNs,
        { revision: state.revision },
        recordHistory,
      );
    summarize(time, recordHistory);
  } else throw Error("Unknown observation page");
  $("delivery").textContent =
    `${receivedFrames} frames · ${bytes(receivedBytes)}`;
  redraw();
}

function state(label, live = false) {
  $("connection").textContent = label;
  $("connection-dot").classList.toggle("live", live);
}

function disconnect() {
  clearTimeout(retry);
  if (!socket) return;
  socket.close();
  socket = undefined;
  for (const ring of Object.values(history)) ring.push(latestTime, NaN);
  for (const model of models.values()) model.ring.push(latestTime, NaN);
  for (const pool of pools.values()) pool.ring.push(latestTime, NaN);
}

function connect() {
  if (paused || socket) return;
  for (const { rate } of serverRates) rate.reset();
  for (const bucket of Object.values(serverBuckets)) bucket.reset();
  for (const model of models.values()) {
    model.requests.rate.reset();
    model.work.rate.reset();
  }
  const url = new URL("../v1/observe", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  $("endpoint").textContent = location.host;
  state("Connecting");
  const peer = new WebSocket(url, "loom.observe.v1");
  socket = peer;
  sequence = 0n;
  peer.onopen = () => {
    if (socket === peer) state("Live", true);
  };
  peer.onmessage = (event) => {
    if (socket !== peer) return;
    try {
      ++receivedFrames;
      receivedBytes += encoder.encode(event.data).length;
      const frame = JSON.parse(event.data);
      apply(frame);
      $("notice").hidden = true;
      retryDelay = 1000;
      // Applied facts and bounded rings, not socket delivery, return credit.
      peer.send(frame.sequence);
    } catch (error) {
      $("notice").textContent = `Observation error: ${error.message}`;
      $("notice").hidden = false;
      paused = true;
      $("pause").textContent = "Resume";
      peer.close(1002, "Invalid observation");
    }
  };
  peer.onclose = () => {
    if (socket !== peer) return;
    disconnect();
    state(paused ? "Paused" : "Disconnected");
    if (!paused) {
      $("notice").textContent =
        "Live feed disconnected. Relays must forward WebSocket upgrades and preserve the public origin. Reconnecting…";
      $("notice").hidden = false;
      retry = setTimeout(connect, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 30000);
    }
  };
}

$("pause").onclick = () => {
  pausedTime = chartTime();
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  if (paused) {
    state("Paused");
    disconnect();
  } else connect();
};
function updateChartAxes() {
  const labels = [4, 3, 2, 1].map((tick) => `−${(windowMs / 1000) * tick / 4}s`);
  labels.push(scrolling ? "NOW" : "LAST ACTIVITY");
  for (const footer of document.querySelectorAll(".chart-footer")) {
    footer.replaceChildren(...labels.map((label) => element("span", "", label)));
  }
}
updateChartAxes();
$("scroll").onclick = () => {
  scrolling = !scrolling;
  $("scroll").setAttribute("aria-pressed", String(scrolling));
  $("scroll").textContent = scrolling ? "Scroll on" : "Scroll off";
  updateChartAxes();
  redraw(true);
};
$("window").onclick = () => {
  windowMs = windowMs === 30000 ? 120000 : 30000;
  $("window").textContent = windowMs === 30000 ? "30 SEC" : "2 MIN";
  updateChartAxes();
  redraw(true);
};
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) redraw(true);
});
window.addEventListener("pagehide", disconnect);
window.addEventListener("pageshow", connect);
new ResizeObserver(() => redraw(true)).observe(document.querySelector("main"));
setInterval(() => {
  if (document.hidden) return;
  $("clock").textContent = new Date().toLocaleTimeString(undefined, {
    hour12: false,
  });
  if (receivedAt)
    $("freshness").textContent =
      `Last sample ${((performance.now() - receivedAt) / 1000).toFixed(1)}s ago`;
  if (scrolling && !paused && receivedAt) redraw(true);
}, 1000);
