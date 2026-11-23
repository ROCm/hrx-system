// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { fileURL, offeredTools } from "./wire.js";
import { imageTypes } from "./content.js";
import { emptySession } from "./storage.js";
import { importSettings } from "./settings.js";

function requireField(condition, name) {
  if (!condition) throw new Error("Invalid conversation export · " + name);
}
const string = (value) => typeof value === "string";
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
function importedSizes(sizes, entries) {
  requireField(object(sizes), "preview sizes");
  for (const [key, size] of Object.entries(sizes)) {
    const address = /^(0|[1-9]\d*)\/(original\/)?(0|[1-9]\d*)$/.exec(key);
    requireField(
      address &&
        Number.isSafeInteger(+address[3]) &&
        +address[1] < entries.length &&
        (!address[2] || entries[address[1]].against),
      "preview position",
    );
    requireField(
      object(size) &&
        Object.keys(size).length &&
        Object.entries(size).every(
          ([axis, value]) =>
            (axis === "width" || axis === "height") && Number.isSafeInteger(value) && value > 0,
        ),
      "preview dimensions",
    );
  }
  return sizes;
}
function importedFile(record) {
  requireField(object(record) && imageTypes.includes(record.type) && string(record.data), "image");
  const prefix = "data:" + record.type + ";base64,";
  requireField(record.data.startsWith(prefix), "embedded image data");
  const bytes = Uint8Array.from(atob(record.data.slice(prefix.length)), (char) =>
    char.charCodeAt(0),
  );
  return record.name === undefined
    ? new Blob([bytes], { type: record.type })
    : new File([bytes], record.name, { type: record.type });
}
function importedMessage(message) {
  requireField(
    object(message) && ["user", "assistant", "tool"].includes(message.role),
    "message role",
  );
  const result = { role: message.role, content: message.content };
  if (Array.isArray(message.content) && message.role === "user") {
    result.content = message.content.map((part) => {
      if (part?.type === "image") return { type: "image", file: importedFile(part.file) };
      requireField(object(part) && string(part.text), "message part");
      if (part.type === "document" && string(part.name))
        return { type: "document", name: part.name, text: part.text };
      requireField(part.type === "text", "message part type");
      return { type: "text", text: part.text };
    });
  } else requireField(string(message.content), "message content");
  if (message.role === "tool") {
    requireField(string(message.tool_call_id) && message.tool_call_id, "tool result ID");
    result.tool_call_id = message.tool_call_id;
  }
  if (message.reasoning_content !== undefined) {
    requireField(string(message.reasoning_content), "reasoning");
    result.reasoning_content = message.reasoning_content;
  }
  if (message.tool_calls !== undefined) {
    requireField(message.role === "assistant" && Array.isArray(message.tool_calls), "tool calls");
    result.tool_calls = message.tool_calls.map((call) => {
      requireField(
        call?.type === "function" &&
          string(call.id) &&
          call.id &&
          string(call.function?.name) &&
          call.function.name &&
          string(call.function.arguments),
        "tool call",
      );
      return {
        id: call.id,
        type: "function",
        function: { name: call.function.name, arguments: call.function.arguments },
      };
    });
  }
  return result;
}
function importedStats(stats) {
  requireField(
    object(stats) && Array.isArray(stats.arrivals) && stats.arrivals.every(Number.isFinite),
    "stream timings",
  );
  for (const key of ["firstText", "firstDelta", "elapsed", "frames"])
    requireField(stats[key] === undefined || Number.isFinite(stats[key]), key);
  if (stats.samples !== undefined) {
    const points = stats.samples;
    requireField(
      Array.isArray(points) &&
        points.length > 0 &&
        points.length <= 65 &&
        points.every(
          (p, i) =>
            Array.isArray(p) &&
            (p.length === 2 || p.length === 3) &&
            p.every((v) => Number.isSafeInteger(v) && v >= 0) &&
            (!i ||
              (p.length === points.at(-1)?.length &&
                p[0] > points[i - 1][0] &&
                p[1] >= points[i - 1][1] &&
                (p[2] ?? 0) >= (points[i - 1][2] ?? 0))),
        ),
      "token samples",
    );
  }
  if (stats.usage != null) {
    const usage = stats.usage,
      cached = usage.prompt_tokens_details?.cached_tokens;
    requireField(
      Number.isSafeInteger(usage.prompt_tokens) &&
        Number.isSafeInteger(usage.completion_tokens) &&
        (cached == null || Number.isSafeInteger(cached)),
      "token usage",
    );
  }
  for (const key of ["model", "finish"])
    requireField(stats[key] === undefined || string(stats[key]), key);
  return stats;
}
export async function conversationImport(file, project) {
  const data = JSON.parse(await file.text());
  requireField(
    data?.format === "loom.conversation.v1" && object(data.session) && Array.isArray(data.entries),
    "format",
  );
  const { title, model, settings } = data.session;
  requireField(string(title) && (model === undefined || string(model)), "session header");
  // The imported transcript is complete. Native session IDs, checkpoints,
  // parent references and revision counters belong to the source browser.
  const session = emptySession(project, importSettings(settings));
  Object.assign(session, { title, named: true, model, importedCount: data.entries.length });
  const pending = new Set();
  for (const input of data.entries) {
    requireField(object(input), "entry");
    const entry = {};
    if (input.message) entry.message = importedMessage(input.message);
    else {
      requireField(["user", "assistant"].includes(input.role), "entry role");
      entry.role = input.role;
    }
    if (input.compaction !== undefined) {
      const prompt = session.entries.at(-1);
      requireField(
        input.compaction === true &&
          entry.message?.role === "assistant" &&
          entry.message.content.trim() &&
          !entry.message.tool_calls &&
          !input.partial &&
          !pending.size &&
          prompt?.message?.role === "user" &&
          !prompt.partial,
        "compaction boundary",
      );
      entry.compaction = true;
    }
    for (const key of ["model", "imagePrompt", "imageModel", "html", "text", "problem", "draft"])
      if (input[key] != null) {
        requireField(string(input[key]), key);
        entry[key] = input[key];
      }
    if (input.toolUpdate !== undefined) {
      requireField(
        entry.message?.role === "user" && string(input.toolUpdate) && input.toolUpdate,
        "tool update",
      );
      entry.toolUpdate = input.toolUpdate;
    }
    for (const key of ["partial", "offeredHTML", "failed", "verbatim"])
      if (input[key] !== undefined) {
        requireField(typeof input[key] === "boolean", key);
        entry[key] = input[key];
      }
    if (input.settings !== undefined) {
      entry.settings = importSettings(input.settings);
      if (entry.message?.role === "assistant") offeredTools(entry.settings);
    }
    if (input.image !== undefined) entry.image = importedFile(input.image);
    if (input.stats !== undefined) entry.stats = importedStats(input.stats);
    if (input.against !== undefined) {
      const source = input.against;
      requireField(
        source?.message?.role === "assistant" &&
          string(source.model) &&
          entry.message?.role === "assistant",
        "comparison",
      );
      entry.against = { message: importedMessage(source.message), model: source.model };
      if (source.stats !== undefined) entry.against.stats = importedStats(source.stats);
      if (string(source.problem)) entry.against.problem = source.problem;
    }
    if (!entry.partial && entry.message) {
      for (const call of entry.message.tool_calls || []) {
        requireField(!pending.has(call.id), "duplicate pending tool ID");
        pending.add(call.id);
      }
      if (entry.message.role === "tool")
        requireField(pending.delete(entry.message.tool_call_id), "unmatched tool result");
    }
    session.entries.push(entry);
  }
  if (data.session.previewSizes !== undefined)
    session.previewSizes = importedSizes(data.session.previewSizes, session.entries);
  return session;
}

async function embeddedFile(file, signal) {
  return { name: file.name, type: file.type, data: await fileURL(file, signal) };
}

function portableSettings(values, history) {
  if (!values?.mcp) return values;
  const { mcp, ...settings } = values;
  if (history)
    settings.functions = JSON.stringify([
      ...mcp.map((b) => ({ type: "function", function: b.fn })),
      ...JSON.parse(settings.functions?.trim() || "[]"),
    ]);
  return settings;
}

async function exportedEntry(entry, signal) {
  const result = { ...entry };
  if (entry.settings) result.settings = portableSettings(entry.settings, true);
  if (entry.image) result.image = await embeddedFile(entry.image, signal);
  if (Array.isArray(entry.message?.content)) {
    const content = [];
    for (const part of entry.message.content) {
      content.push(
        part.type === "image" ? { ...part, file: await embeddedFile(part.file, signal) } : part,
      );
    }
    result.message = { ...entry.message, content };
  }
  return result;
}

export async function conversationDownload(session, signal) {
  const { id, title, model, settings, parent, previewSizes } = session;
  const parts = [
    '{\n  "format": "loom.conversation.v1",\n  "session": ',
    JSON.stringify(
      {
        id,
        title,
        model,
        settings: portableSettings(settings, false),
        parent,
        previewSizes: previewSizes && Object.keys(previewSizes).length ? previewSizes : undefined,
      },
      null,
      2,
    ),
    ',\n  "entries": [',
  ];
  // Keep base64 strings for only one entry at a time. The final Blob can
  // reference already encoded parts without stringifying the whole transcript.
  for (const [index, entry] of session.entries.entries()) {
    signal.throwIfAborted();
    const record = await exportedEntry(entry, signal);
    parts.push(new Blob([index ? ",\n" : "\n", JSON.stringify(record, null, 2)]));
  }
  signal.throwIfAborted();
  parts.push("\n  ]\n}\n");
  const link = document.createElement("a");
  link.download = "loom-conversation-" + id + ".json";
  link.href = URL.createObjectURL(new Blob(parts, { type: "application/json" }));
  link.textContent = "Download JSON";
  return link;
}
