// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { chartColors } from "./topbar.js";
import { byId, element } from "./view.js";
import { checked, endpoint } from "./wire.js";

const rate = (tokens, ms) => (ms > 0 ? (1000 * tokens) / ms : null);
const seconds = (ms) => (ms / 1000).toFixed(ms < 60000 ? 1 : 0) + " s";

function drawArrivals(canvas, stats) {
  const ctx = canvas.getContext("2d"),
    { width, height } = canvas,
    base = height - 4;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = chartColors.rule;
  ctx.fillRect(0, base, width, 2);
  ctx.fillStyle = chartColors.output;
  for (const time of stats.arrivals || [])
    ctx.fillRect(
      ((width - 2) * time) / Math.max(1, stats.elapsed),
      height / 3,
      2,
      base - height / 3,
    );
}

// The time axis spans the turn. Each phase scales its own rates so quick
// prefill cannot flatten the output curve. A mixed interval has no phase rate.
function drawRates(canvas, samples, latest) {
  const points = latest && latest[0] > samples.at(-1)[0] ? [...samples, latest] : samples;
  const ctx = canvas.getContext("2d"),
    { width, height } = canvas,
    base = height - 4;
  const time = (ms) => 1 + ((width - 4) * ms) / Math.max(1, points.at(-1)[0]);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = chartColors.rule;
  ctx.fillRect(0, base, width, 2);
  function trace(series, field, color) {
    if (series.length < 2) return;
    const speeds = series
      .slice(1)
      .map((point, i) => rate(point[field] - series[i][field], point[0] - series[i][0]));
    const peak = Math.max(1, ...speeds),
      y = (speed) => base - ((base - 4) * speed) / peak;
    const start = time(series[0][0]),
      end = time(series.at(-1)[0]);
    ctx.beginPath();
    ctx.moveTo(start, y(speeds[0]));
    speeds.forEach((speed, i) => ctx.lineTo(time(series[i + 1][0]), y(speed)));
    ctx.lineJoin = "round";
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.lineTo(end, base);
    ctx.lineTo(start, base);
    ctx.closePath();
    ctx.globalAlpha = 0.17;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(end, y(speeds.at(-1)), 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }
  if (points.at(-1).length === 2) {
    trace(points, 1, chartColors.output);
    return;
  }
  const first = points.findIndex((p) => p[1] > 0);
  trace(points.slice(1, first < 0 ? points.length : first), 2, chartColors.input);
  if (first >= 0) trace(points.slice(first), 1, chartColors.output);
}

function meter(stats) {
  const node = element("footer", undefined, "reply-stats"),
    canvas = element("canvas");
  canvas.width = 184;
  canvas.height = 36;
  canvas.setAttribute("role", "img");
  const phase = element("span", undefined, "reply-phase"),
    count = element("span", undefined, "reply-count");
  const speed = element("span", undefined, "reply-rate");
  const clock = element("span"),
    cache = element("span", undefined, "reply-cache");
  node.append(canvas, phase, count, speed, clock, cache);
  function paint(live) {
    const usage = stats.usage,
      elapsed = stats.elapsed ?? 0;
    const tokens = usage?.completion_tokens ?? live?.latest?.[1] ?? stats.samples?.at(-1)?.[1];
    const text = Boolean(live?.text || usage || stats.arrivals?.length || stats.samples);
    const current = live ? live.speed : usage && rate(usage.completion_tokens, elapsed);
    const prompt = live?.prompt,
      prefill = live?.phase === "Prefilling";
    node.toggleAttribute("data-live", !!live);
    phase.textContent = live?.phase || "";
    phase.style.color = prefill ? chartColors.input : "";
    clock.textContent = seconds(elapsed);
    clock.title = "Elapsed since the request started";
    count.textContent = prefill
      ? (prompt.cached_tokens + prompt.processed_tokens).toLocaleString() +
        " / " +
        prompt.total_tokens.toLocaleString() +
        " prompt"
      : text
        ? (tokens == null ? "—" : tokens.toLocaleString()) + " tok"
        : stats.frames != null
          ? stats.frames.toLocaleString() + " frames"
          : "";
    count.title = prefill
      ? "Cached and processed prompt tokens / total prompt tokens"
      : "Output tokens";
    speed.textContent =
      text && (live || current != null)
        ? (current == null ? "—" : current.toFixed(1)) + (live ? " tok/s" : " tok/s avg")
        : "";
    speed.title = prefill
      ? "Fresh prompt tokens per second between completed batches"
      : live
        ? "Output rate between samples"
        : "Average output rate including prompt processing and wait time";
    const cached = usage?.prompt_tokens_details?.cached_tokens ?? prompt?.cached_tokens;
    cache.textContent = cached ? cached.toLocaleString() + " cached" : "";
    if (usage)
      node.title =
        usage.prompt_tokens.toLocaleString() +
        " prompt tokens" +
        (cached == null ? "" : " · " + cached.toLocaleString() + " cached");
    else node.title = live?.error ? "Live token counters unavailable · " + live.error : "";
    canvas.hidden = !text;
    if (!text) return;
    const label =
      stats.samples?.at(-1)?.length === 3
        ? "Fresh prompt and output tokens per second, scaled separately"
        : stats.samples
          ? "Sampled output tokens per second"
          : "Stream event arrivals over elapsed time";
    canvas.title = label;
    canvas.setAttribute("aria-label", label);
    if (stats.samples) drawRates(canvas, stats.samples, live?.latest);
    else drawArrivals(canvas, stats);
  }
  return { node, paint };
}

export function savedReplyStats(stats) {
  const view = meter(stats);
  view.paint();
  return view.node;
}

// The role chunk's loom-N and the active request inventory share a receipt
// serial within this model. A session can span many receipts, or be anonymous.
async function requestSnapshot(model, id, signal) {
  const headers = { "X-Loom-Model": model };
  let offset = 0;
  do {
    if (offset) headers["X-Loom-Offset"] = String(offset);
    const reply = await checked(
      await fetch(endpoint("/v1/requests"), { headers, signal, cache: "no-store" }),
    );
    const page = await reply.json();
    const row = page.data.find((row) => Number.isSafeInteger(row.id) && "loom-" + row.id === id);
    if (row) return row;
    offset = page.next_offset;
  } while (offset != null);
}

export function watchReplyStats(stats, start, kind, inspect) {
  const view = meter(stats),
    scope = new AbortController();
  const live = {
    phase: kind === "text" ? "Waiting" : kind === "image" ? "Generating image" : "Playing audio",
    text: kind === "text",
  };
  let id,
    sampling = false,
    retryAt = 0,
    ordinal = 0,
    stride = 1;
  function retain(point, final) {
    const samples = (stats.samples ??= [[0, 0]]);
    if (!final && ++ordinal % stride) return;
    if (point[0] > samples.at(-1)[0]) samples.push(point);
    else if (point[0] === samples.at(-1)[0]) samples[samples.length - 1] = point;
    // Fold adjacent intervals in pairs so even long generations retain both ends.
    if (samples.length === 65) {
      stats.samples = samples.filter((_, i) => !(i % 2));
      stride *= 2;
    }
  }
  function paint() {
    stats.elapsed = performance.now() - start;
    view.paint(live);
    inspect(stats, live.latest?.[1], live.prompt);
  }
  async function tick() {
    if (!id || sampling || performance.now() < retryAt) {
      paint();
      return;
    }
    sampling = true;
    try {
      const row = await requestSnapshot(stats.model, id, scope.signal);
      if (!row || scope.signal.aborted) return;
      delete live.error;
      const { prompt, counters } = row,
        output = counters.output_tokens;
      const next = [Math.round(performance.now() - start), output],
        previous = live.latest;
      if (prompt) next.push(prompt.processed_tokens);
      live.prompt = prompt;
      const prefill =
        prompt &&
        !output &&
        !live.tool &&
        live.phase !== "Generating" &&
        prompt.cached_tokens + prompt.processed_tokens < prompt.total_tokens;
      if (!live.tool && (prefill || output || prompt))
        live.phase = prefill ? "Prefilling" : "Generating";
      // An unchanged prompt frontier is a batch in flight. Measure elapsed
      // time between advances, even when several polls see the same frontier.
      if (prefill && previous?.[2] === next[2]) return;
      const field = prefill ? 2 : 1;
      live.speed =
        previous && (prefill || previous[1] > 0)
          ? rate(next[field] - previous[field], next[0] - previous[0])
          : null;
      live.latest = next;
      retain(next, false);
    } catch (error) {
      live.error = error.message;
      live.speed = null;
      retryAt = [401, 403, 404, 405].includes(error.status) ? Infinity : performance.now() + 2000;
    } finally {
      sampling = false;
      if (!scope.signal.aborted) paint();
    }
  }
  paint();
  const timer = setInterval(tick, 500);
  return {
    node: view.node,
    stream(receipt, delta, tool) {
      if (id === undefined) id = /^loom-[0-9]+$/.test(receipt) ? receipt : null;
      if (live.phase === "Prefilling" && (tool || delta?.content || delta?.reasoning_content))
        live.speed = null;
      if (tool) {
        live.phase = "Calling " + tool;
        live.tool = true;
      } else if (delta?.content || delta?.reasoning_content) live.phase = "Generating";
      else if (live.phase === "Waiting") live.phase = "Processing";
    },
    end() {
      if (scope.signal.aborted) return;
      clearInterval(timer);
      scope.abort();
      stats.elapsed = performance.now() - start;
      if (stats.samples) {
        let last = live.latest;
        if (stats.usage) {
          last = [Math.round(stats.elapsed), stats.usage.completion_tokens];
          if (live.prompt) last.push(live.prompt.total_tokens - live.prompt.cached_tokens);
        }
        retain(last, true);
      }
      view.paint();
    },
  };
}

export function metrics(stats, liveOutput, promptProgress) {
  byId("turn-inspector").hidden = !stats;
  if (!stats) return;
  const ms = (value) => (value == null ? "—" : Math.round(value) + " ms");
  const usage = stats.usage;
  const rows = [
    ["Model", stats.model],
    ["First output delta", ms(stats.firstDelta ?? stats.firstText)],
    ["Elapsed", ms(stats.elapsed)],
    ["Prompt tokens", usage?.prompt_tokens ?? promptProgress?.total_tokens],
    ["Cached prompt", usage?.prompt_tokens_details?.cached_tokens ?? promptProgress?.cached_tokens],
    ["Output tokens", usage?.completion_tokens ?? liveOutput ?? stats.samples?.at(-1)?.[1]],
    ["Finish", stats.finish],
  ];
  if (stats.frames != null) rows.push(["Playback frames", stats.frames]);
  byId("metrics").replaceChildren(
    ...rows.flatMap(([name, value]) => [element("dt", name), element("dd", value ?? "—")]),
  );
  drawArrivals(byId("trace"), stats);
}
export function drawTurns(entries) {
  const turns = entries.filter((entry) => entry.stats?.usage);
  const visible = turns.slice(-16);
  byId("turn-history").hidden = !visible.length;
  const bars = byId("turns");
  bars.replaceChildren();
  const ceiling = Math.max(
    1,
    ...visible.map(({ stats }) => stats.usage.prompt_tokens + stats.usage.completion_tokens),
  );
  for (const [i, { stats }] of visible.entries()) {
    const { prompt_tokens: input, completion_tokens: output, prompt_tokens_details } = stats.usage;
    const cached = prompt_tokens_details?.cached_tokens;
    const button = element("button");
    const label =
      "Turn " +
      (turns.length - visible.length + i + 1) +
      " · " +
      input +
      " prompt / " +
      (cached ?? "unknown") +
      " cached / " +
      output +
      " output";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", String(i === visible.length - 1));
    const canvas = element("canvas");
    canvas.width = 24;
    canvas.height = 60;
    const context = canvas.getContext("2d");
    let y = 60;
    for (const [value, color] of [
      [input - (cached ?? 0), cached == null ? chartColors.dim : chartColors.input],
      [cached ?? 0, chartColors.memory],
      [output, chartColors.output],
    ]) {
      const height = (value * 60) / ceiling;
      context.fillStyle = color;
      context.fillRect(1, y - height, 22, height);
      y -= height;
    }
    button.append(canvas);
    button.onclick = () => {
      for (const peer of bars.children) peer.setAttribute("aria-pressed", String(peer === button));
      metrics(stats);
    };
    bars.append(button);
  }
}
