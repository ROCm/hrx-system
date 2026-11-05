// Copyright 2026 The IREE Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  constructor(capacity = 512) {
    this.times = new Float64Array(capacity);
    this.values = new Float64Array(capacity);
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
const history = Object.fromEntries(
  ["output", "input", "requests", "arrivals", "finishes", "kv", "errors"].map(
    (key) => [key, new Ring()],
  ),
);
const models = new Map();
const devices = new Map();
const pools = new Map();
const pendingPlots = new Set();
const serverRates = Object.fromEntries(
  ["output_tokens", "input_tokens", "completed"].map((key) => [
    key,
    new CounterRate(),
  ]),
);
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
let dirty = false;
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
  for (const [ring] of series) {
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
  for (const [ring, color] of series) {
    const points = [];
    let lastTime = 0;
    let lastValue = NaN;
    ring.each((time, value) => {
      if (time >= now - windowMs && time <= now) points.push([time, value]);
      if (time <= now) {
        lastTime = time;
        lastValue = value;
      }
    });
    const x = (time) =>
      left + ((time - now + windowMs) / windowMs) * (width - left);
    const y = (value) => bottom - (value / maximum) * (bottom - top);
    const gradient = context.createLinearGradient(0, top, 0, bottom);
    gradient.addColorStop(0, `${color}35`);
    gradient.addColorStop(1, `${color}00`);
    // Missing observations create real gaps, not invented interpolated work.
    let segment = [];
    const paint = () => {
      if (!segment.length) return;
      context.beginPath();
      context.moveTo(x(segment[0][0]), bottom);
      for (const [time, value] of segment) context.lineTo(x(time), y(value));
      context.lineTo(x(segment.at(-1)[0]), bottom);
      context.closePath();
      context.fillStyle = gradient;
      context.fill();
      context.beginPath();
      segment.forEach(([time, value], index) =>
        index
          ? context.lineTo(x(time), y(value))
          : context.moveTo(x(time), y(value)),
      );
      context.lineWidth = grid ? 1.7 : 1.4;
      context.strokeStyle = color;
      context.stroke();
      const last = segment.at(-1);
      context.beginPath();
      context.arc(x(last[0]), y(last[1]), grid ? 2.8 : 2, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      segment = [];
    };
    for (const point of points) {
      if (!Number.isFinite(point[1])) {
        paint();
        continue;
      }
      if (segment.length && point[0] - segment.at(-1)[0] > 2000) paint();
      segment.push(point);
    }
    paint();
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
  dirty = true;
  if (all) {
    for (const model of models.values()) pendingPlots.add(model);
    for (const pool of pools.values()) pendingPlots.add(pool);
  }
  if (framePending || document.hidden) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    if (!dirty) return;
    dirty = false;
    const now = chartTime();
    draw(
      $("flow"),
      [
        [history.input, COLORS.input],
        [history.output, COLORS.output],
      ],
      now,
      true,
    );
    draw($("request-flow"), [[history.requests, COLORS.input]], now, true);
    draw(
      $("activity-flow"),
      [
        [history.arrivals, COLORS.active],
        [history.finishes, COLORS.input],
      ],
      now,
      true,
    );
    for (const [id, key, color] of [
      ["output-spark", "output", COLORS.output],
      ["input-spark", "input", COLORS.input],
      ["request-spark", "requests", COLORS.input],
      ["activity-spark", "arrivals", COLORS.active],
      ["kv-spark", "kv", COLORS.memory],
      ["error-spark", "errors", COLORS.errors],
    ])
      draw($(id), [[history[key], color]], now);
    for (const model of pendingPlots)
      draw(model.canvas, [[model.ring, model.color]], now);
    pendingPlots.clear();
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
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

function deviceLabel(device) {
  return device.key + (device.key_truncated ? "…" : "");
}

function updatePlacementLinks(container, indices, inventory, kind, label) {
  if (!indices.length) {
    container.textContent = `unassigned ${kind}`;
    return;
  }
  if (container.children.length !== indices.length) {
    container.replaceChildren(
      ...indices.map(() => element("a", "device-link")),
    );
  }
  indices.forEach((index, position) => {
    const resource = inventory.get(index);
    const link = container.children[position];
    link.textContent = resource ? label(resource.source) : `${kind} ${index}`;
    if (resource) link.href = `#${kind}-${index}`;
    else link.removeAttribute("href");
  });
}

function updatePlacement(state) {
  updatePlacementLinks(
    state.placement, state.model.devices, devices, "device", deviceLabel,
  );
  updatePlacementLinks(
    state.poolPlacement, state.model.pools, pools, "pool", poolLabel,
  );
}

function summarizeDevices() {
  const activity = new Map(
    [...devices.keys()].map((index) => [
      index,
      { models: 0, active: 0, queued: 0, held: 0 },
    ]),
  );
  for (const state of models.values()) {
    updatePlacement(state);
    for (const index of state.model.devices) {
      const totals = activity.get(index);
      if (!totals) continue;
      ++totals.models;
      totals.active += state.model.active;
      totals.queued += state.model.queued;
      totals.held += state.model.held;
    }
  }
  for (const [index, totals] of activity) {
    devices.get(index).activity.textContent =
      `${totals.models} models · ${totals.active} active · ${totals.queued} queued · ${totals.held} held`;
  }
}

function poolLabel(pool) {
  return pool.key + (pool.key_truncated ? "…" : "");
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
        device ? deviceLabel(device.source) : `device ${index}`,
      );
      link.href = `#device-${index}`;
      pool.links.append(link);
      if (device) {
        const owner = element("a", "device-link", poolLabel(pool.source));
        owner.href = `#pool-${pool.source.index}`;
        device.links.append(owner);
      }
    }
  }
}

function createPool(source) {
  const card = element("article", "panel memory-pool");
  card.id = `pool-${source.index}`;
  const heading = element("div", "resource-head");
  const links = element("div", "device-pools");
  heading.append(
    element("h3", "", poolLabel(source)),
    element("span", "muted", source.backing),
    links,
  );
  const total = element("div", "pool-total");
  const committed = element("strong", "pool-value", "—");
  const peak = element("span", "muted");
  peak.title = "Highest commitment observed in this page session.";
  total.append(committed, peak);
  const canvas = element("canvas");
  canvas.setAttribute(
    "aria-label",
    `${poolLabel(source)} committed bytes`,
  );
  const budget = element("div", "model-facts");
  const reservations = element("div", "model-facts");
  const workspace = element("div", "model-facts");
  workspace.title =
    "Workspace is backed by this pool. It is not additional physical usage; atomic samples need not agree at the same instant.";
  const details = element("details");
  const table = element("table");
  details.append(element("summary", "", "All pool counters"), table);
  card.append(heading, total, canvas, budget, reservations, workspace, details);
  $("pool-grid").append(card);
  return {
    source,
    links,
    card,
    committed,
    peak,
    canvas,
    budget,
    reservations,
    workspace,
    table,
    cells: new Map(),
    ring: new Ring(),
    color: COLORS.memory,
  };
}

function updateDevice(device) {
  let state = devices.get(device.index);
  if (!state) {
    if (!devices.size) $("device-grid").replaceChildren();
    const card = element("article", "panel device");
    card.id = `device-${device.index}`;
    const heading = element("div", "resource-head");
    const identity = element(
      "span",
      "device-identity",
      `${device.name} · ${device.driver} / ${device.backend}`,
    );
    heading.append(
      element("h3", "", deviceLabel(device)),
      identity,
    );
    const activity = element("div", "device-activity");
    const links = element("div", "device-pools");
    card.append(heading, activity, links);
    $("device-grid").append(card);
    state = { source: device, card, activity, links };
    devices.set(device.index, state);
    updateResourceLinks();
  }
  $("device-count").textContent = devices.size;
  summarizeDevices();
}

function updatePool(source, time, recordHistory) {
  let pool = pools.get(source.index);
  if (!pool) {
    if (!pools.size) $("pool-grid").replaceChildren();
    pool = createPool(source);
    pools.set(source.index, pool);
    updateResourceLinks();
  }
  const fields = updateTopicFields(source.topics, pool);
  const value = (topic, name) =>
    fields.has(`${topic}.${name}`)
      ? Number(fields.get(`${topic}.${name}`).value)
      : NaN;
  const elastic = source.backing === "elastic";
  const storageBytes = value("storage", "committed_bytes");
  const elasticBytes = elastic ? value("elastic", "committed_bytes") : 0;
  // Allocator breakdowns are included in the physical owner's total.
  const committed = elastic ? elasticBytes : storageBytes;
  pool.committed.textContent = bytes(committed);
  if (recordHistory) {
    pool.ring.push(time, committed);
    pendingPlots.add(pool);
  }
  pool.peak.textContent = `peak ${bytes(pool.ring.peak)}`;
  const limit = elastic ? value("elastic", "budget_limit") : 0;
  pool.budget.textContent = elastic
    ? `${bytes(elasticBytes)} elastic · ${bytes(limit)} budget · ${bytes(storageBytes)} persistent allocator (included)`
    : "Native storage · no pool budget";
  pool.reservations.textContent = elastic
    ? `${bytes(value("elastic", "reserved_bytes"))} virtual reserved · ${bytes(value("elastic", "cached_bytes"))} reusable`
    : `${bytes(value("storage", "reserved_bytes"))} charged · ${bytes(value("storage", "free_bytes"))} reusable`;
  pool.workspace.textContent =
    `Workspace: ${bytes(value("workspace", "reserved_bytes"))} charged · ${bytes(value("workspace", "committed_bytes"))} retained (included)`;
  $("pool-count").textContent = pools.size;
  summarizeDevices();
}

function createModel(model) {
  if (!models.size) $("model-grid").replaceChildren();
  const card = element("article", "panel model");
  const heading = element("div", "model-head");
  const name = element(
    "h3",
    "",
    model.name + (model.name_truncated ? "…" : ""),
  );
  const identity = element("div");
  const kind = element("div", "kind", model.kind);
  const placement = element("span");
  const poolPlacement = element("span");
  kind.append(placement, poolPlacement);
  identity.append(name, kind);
  const phase = element("span", "phase", model.phase);
  heading.append(identity, phase);
  const metrics = element("div", "model-metrics");
  const workUnit =
    model.kind === "image"
      ? "Images"
      : model.kind === "text"
        ? "Tokens"
        : "Batches";
  const numbers = {};
  const peaks = {};
  const averages = {};
  for (const [key, label] of Object.entries({
    active: "Active now",
    queued: "Queued + held",
    completed: "Completed",
    requests: "Requests/s · 5s",
    work: `${workUnit}/s · 5s`,
    total: `${workUnit} total`,
  })) {
    const item = element("div");
    const value = element("strong", "", "—");
    item.append(element("label", "", label), value);
    if (key !== "completed" && key !== "total") {
      const peak = element("span", "peak", "peak —");
      peak.title =
        "Observed peak in this page session; rates require a full five-second window.";
      item.append(peak);
      peaks[key] = peak;
    }
    if (key === "work" || key === "requests") {
      const average = element("span", "average", "avg —");
      average.title =
        "Lifetime average, including idle time; independent of this page session.";
      item.append(average);
      averages[key] = average;
    }
    metrics.append(item);
    numbers[key] = value;
  }
  const canvas = element("canvas");
  canvas.setAttribute("aria-label", `${model.name} completed work rate`);
  const pool = element("div");
  const poolHeading = element("div", "pool-heading");
  const poolValue = element("span", "", "—");
  poolHeading.append(element("span", "", "KV positions"), poolValue);
  const bar = element("div", "bar");
  const resident = element("span");
  const reserved = element("span", "reserved");
  bar.append(resident, reserved);
  pool.append(poolHeading, bar);
  pool.hidden = true;
  const cache = element("div", "model-facts");
  const outcomes = element("div", "model-facts");
  const details = element("details");
  const table = element("table");
  details.append(element("summary", "", "All source counters"), table);
  card.append(heading, metrics, canvas, pool, cache, outcomes, details);
  $("model-grid").append(card);
  return {
    id: model.incarnation,
    card,
    phase,
    placement,
    poolPlacement,
    numbers,
    peaks,
    averages,
    activePeak: 0,
    queuedPeak: 0,
    canvas,
    pool,
    poolValue,
    resident,
    reserved,
    cache,
    outcomes,
    table,
    cells: new Map(),
    ring: new Ring(512),
    requestRate: new CounterRate(),
    workRate: new CounterRate(),
    color: model.kind === "image" ? COLORS.input : COLORS.output,
  };
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
  const value = (key) =>
    fields.has(key) ? Number(fields.get(key).value) : NaN;
  const imageTotal = fields.get("image.generated_images")?.value;
  const workTotal =
    model.kind === "image"
      ? imageTotal
      : model.counters?.[model.kind === "text" ? "output_tokens" : "batches"];
  const unchangedUntil = unchangedBoundary(state.revision, version, timeNs);
  const workRate = state.workRate.push(timeNs, workTotal, unchangedUntil);
  state.requests = state.requestRate.push(
    timeNs,
    model.counters?.completed,
    unchangedUntil,
  );
  state.revision = version.revision;
  if (recordHistory) {
    state.ring.push(time, workRate);
    pendingPlots.add(state);
  }
  state.model = model;
  state.fields = fields;
  state.phase.textContent = model.phase;
  state.numbers.active.textContent = format(model.active);
  state.numbers.queued.textContent = format(model.queued + model.held);
  state.activePeak = Math.max(state.activePeak, model.active);
  state.queuedPeak = Math.max(state.queuedPeak, model.queued + model.held);
  for (const [key, value] of Object.entries({
    active: state.activePeak,
    queued: state.queuedPeak,
    work: state.workRate.peak,
    requests: state.requestRate.peak,
  }))
    state.peaks[key].textContent = `peak ${format(value)}`;
  state.numbers.completed.textContent = model.counters
    ? integer(model.counters.completed)
    : "—";
  state.numbers.requests.textContent = format(state.requests);
  state.numbers.work.textContent = format(workRate);
  state.averages.work.textContent = `avg ${formatAverage(lifetimeRate(timeNs, model.since_ns, workTotal))}`;
  state.averages.requests.textContent = `avg ${formatAverage(lifetimeRate(timeNs, model.since_ns, model.counters?.completed))}`;
  state.numbers.total.textContent =
    workTotal === undefined ? "—" : integer(workTotal);
  const capacity = value("text.pool_capacity");
  state.pool.hidden = !(capacity > 0);
  if (capacity > 0) {
    const resident = value("text.pool_resident");
    const reserved = value("text.pool_reserved");
    state.poolValue.textContent = `${format(resident)} resident + ${format(reserved)} reserved / ${format(capacity)}`;
    state.resident.style.width = `${(100 * resident) / capacity}%`;
    state.reserved.style.width = `${(100 * reserved) / capacity}%`;
  }
  const facts =
    model.kind === "text"
      ? [
          ["text.session_capacity", "rows"],
          ["text.session_cache_endpoints", "cached sessions"],
          ["text.prefix_cache_endpoints", "cached prefixes"],
        ]
      : model.kind === "image"
        ? [["image.generated_pixels", "pixels generated"]]
        : [];
  state.cache.textContent = facts
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
    if (Number.isFinite(state.requests)) {
      group.rate += state.requests;
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
  const output = serverRates.output_tokens.push(
    timeNs,
    page.counters?.output_tokens,
    unchangedUntil,
  );
  const input = serverRates.input_tokens.push(
    timeNs,
    page.counters?.input_tokens,
    unchangedUntil,
  );
  const requests = serverRates.completed.push(
    timeNs,
    page.counters?.completed,
    unchangedUntil,
  );
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
  if (recordHistory) {
    history.output.push(time, output);
    history.input.push(time, input);
    history.requests.push(time, requests);
    history.errors.push(
      time,
      page.counters ? Number(page.counters.failed) : NaN,
    );
  }
  $("output-rate").textContent = format(output);
  $("input-rate").textContent = format(input);
  $("request-rate").textContent = format(requests);
  $("output-peak").textContent = format(serverRates.output_tokens.peak);
  $("input-peak").textContent = format(serverRates.input_tokens.peak);
  $("request-peak").textContent = format(serverRates.completed.peak);
  for (const [id, key] of [
    ["output", "output_tokens"],
    ["input", "input_tokens"],
    ["request", "completed"],
  ])
    $(`${id}-average`).textContent = formatAverage(
      lifetimeRate(timeNs, page.since_ns, page.counters?.[key]),
    );
  for (const [id, key] of [
    ["completed", "completed"],
    ["output-total", "output_tokens"],
    ["input-total", "input_tokens"],
    ["errors", "failed"],
    ["rejected", "rejected"],
    ["cancelled", "cancelled"],
    ["batches", "batches"],
  ]) {
    $(id).textContent = page.counters ? integer(page.counters[key]) : "—";
  }
}

function apply(frame) {
  if (frame.version !== 1 || BigInt(frame.sequence) !== sequence + 1n)
    throw Error("Unsupported or discontinuous observation stream");
  sequence = BigInt(frame.sequence);
  if (serverIdentity !== frame.server) {
    serverIdentity = frame.server;
    serverRevision = undefined;
    serverSample = undefined;
    for (const key of Object.keys(serverRates))
      serverRates[key] = new CounterRate();
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
    for (const key of Object.keys(history)) history[key] = new Ring();
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
    Object.values(serverRates).some((rate) => rate.count && rate.value !== 0) ||
    Object.values(serverBuckets).some(
      (bucket) => Number.isFinite(bucket.value) && bucket.value !== 0,
    ) ||
    [...models.values()].some(
      (model) =>
        (model.workRate.count && model.workRate.value !== 0) ||
        (model.requestRate.count && model.requestRate.value !== 0),
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
    updatePool(page, time, recordHistory);
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

function connect() {
  if (paused || document.hidden) return;
  clearTimeout(retry);
  for (const rate of Object.values(serverRates)) rate.reset();
  for (const bucket of Object.values(serverBuckets)) bucket.reset();
  for (const model of models.values()) {
    model.requestRate.reset();
    model.workRate.reset();
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
    for (const ring of Object.values(history)) ring.push(latestTime, NaN);
    for (const model of models.values()) model.ring.push(latestTime, NaN);
    for (const pool of pools.values()) pool.ring.push(latestTime, NaN);
    state(paused ? "Paused" : "Disconnected");
    if (!paused && !document.hidden) {
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
  clearTimeout(retry);
  if (paused) {
    state("Paused");
    socket?.close(1000, "Viewer paused");
  } else connect();
};
$("scroll").onclick = () => {
  scrolling = !scrolling;
  $("scroll").setAttribute("aria-pressed", String(scrolling));
  $("scroll").textContent = scrolling ? "Scroll on" : "Scroll off";
  document.querySelectorAll(".chart-footer").forEach((footer) => {
    footer.lastElementChild.textContent = scrolling ? "NOW" : "LAST ACTIVITY";
  });
  redraw(true);
};
$("window").onclick = () => {
  windowMs = windowMs === 30000 ? 120000 : 30000;
  $("window").textContent = windowMs === 30000 ? "30 SEC" : "2 MIN";
  document.querySelectorAll(".chart-footer").forEach((footer) =>
    footer.querySelectorAll("span").forEach((tick, index) => {
      tick.textContent =
        index === 4
          ? scrolling
            ? "NOW"
            : "LAST ACTIVITY"
          : `−${((windowMs / 1000) * (4 - index)) / 4}s`;
    }),
  );
  redraw(true);
};
document.addEventListener("visibilitychange", () => {
  clearTimeout(retry);
  if (document.hidden) socket?.close(1000, "Viewer hidden");
  else if (!paused) {
    connect();
    redraw();
  }
});
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
connect();
