// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { events } from "./events.js";
import { documentText } from "./content.js";

// Endpoint paths are resolved against the server prefix, including Unix relays.
export function endpoint(path) {
  return new URL(".." + path, document.baseURI);
}

export async function checked(response) {
  if (response.ok) return response;
  const body = await response.text();
  let detail = body;
  if (response.headers.get("Content-Type")?.includes("json")) {
    detail = JSON.parse(body).error?.message || body;
  }
  throw Object.assign(new Error(response.status + " " + (detail || response.statusText)), {
    status: response.status,
  });
}

export function post(path, payload, signal, headers = {}) {
  return fetch(endpoint(path), {
    method: "POST",
    cache: "no-store",
    signal,
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(payload),
  }).then(checked);
}

export async function checkpoint(method, model, name, source) {
  const headers = { "X-Loom-Model": model };
  if (source) headers["X-Loom-Session"] = source;
  const response = await fetch(endpoint("/v1/checkpoints/" + encodeURIComponent(name)), {
    method,
    headers,
  });
  if (method === "DELETE" && response.status === 404) return;
  return (await checked(response)).json();
}

export async function* chatChunks(response) {
  if (!response.headers.get("Content-Type")?.startsWith("text/event-stream")) {
    throw new Error("Expected a chat event stream");
  }
  for await (const data of events(response.body)) {
    if (data === "[DONE]") return;
    const chunk = JSON.parse(data);
    if (chunk.error) throw new Error(chunk.error.message);
    yield chunk;
  }
  throw new Error("Connection ended before the final chat event");
}

export function appendCalls(calls, updates) {
  for (const update of updates) {
    const index = update.index;
    if (!Number.isSafeInteger(index) || index < 0 || index > calls.length) {
      throw new Error("Invalid tool call index");
    }
    const call = (calls[index] ??= {
      id: "",
      type: "function",
      function: { name: "", arguments: "" },
    });
    if (update.id) call.id = update.id;
    if (update.type && update.type !== "function") throw new Error("Unsupported tool type");
    call.function.name += update.function?.name || "";
    call.function.arguments += update.function?.arguments || "";
  }
}

function textTool(name, description, argument) {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: {
        type: "object",
        properties: { [argument]: { type: "string" } },
        required: [argument],
        additionalProperties: false,
      },
    },
  };
}
export const htmlTool = textTool(
  "present_html",
  "Present a self-contained HTML answer directly to the user. Embed all assets. Inline CSS, JavaScript, onclick, SVG, and canvas work offline.",
  "html",
);
export const imageTool = textTool(
  "generate_image",
  "Generate and display an image using the user's local image model. Describe the desired image in the prompt.",
  "prompt",
);
export const codeTool = textTool(
  "run_javascript",
  "Compute a result in a fresh, offline JavaScript worker. Supply an async function body as code. Use return for the result, or console.log for output. No DOM, network, or files.",
  "code",
);

export function offeredTools(values) {
  const tools = [];
  if (values.htmlTool) tools.push(htmlTool);
  if (values.codeTool) tools.push(codeTool);
  if (values.imageModel) tools.push(imageTool);
  if (values.mcp?.length || values.functions?.trim()) {
    const extra = values.functions?.trim() ? JSON.parse(values.functions) : [];
    if (!Array.isArray(extra)) throw new Error("Function schemas must be a JSON array");
    const names = new Set(tools.map((tool) => tool.function.name));
    for (const tool of [
      ...(values.mcp || []).map((b) => ({ type: "function", function: b.fn })),
      ...extra,
    ]) {
      const name = tool?.function?.name;
      if (tool?.type !== "function" || typeof name !== "string" || !name.trim())
        throw new Error("Each function schema needs a function name");
      if (names.has(name)) throw new Error("Duplicate offered function · " + name);
      names.add(name);
      tools.push(tool);
    }
  }
  return tools;
}

export function toolChange(previous, tools) {
  if (!previous) return "";
  const names = (list) => list.map((tool) => tool.function.name);
  const before = names(offeredTools(previous)),
    after = names(tools);
  const added = after.filter((name) => !before.includes(name));
  const removed = before.filter((name) => !after.includes(name));
  if (!added.length && !removed.length) return "";
  return [
    "Tool availability changed.",
    added.length && "Enabled: " + added.join(", ") + ".",
    removed.length && "Disabled: " + removed.join(", ") + ".",
  ]
    .filter(Boolean)
    .join(" ");
}

export function toolArgument(call, key) {
  const source = call.function.arguments;
  const args = JSON.parse(source);
  // These tools accept one string. Checking the member count on the wire also
  // catches duplicate keys which JSON.parse collapses into one property.
  const oneStringMember = /^\s*\{\s*"(?:[^"\\]|\\.)*"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}\s*$/s;
  if (!oneStringMember.test(source) || !Object.hasOwn(args, key) || !args[key].trim()) {
    throw new Error(call.function.name + " requires one nonempty string named " + key);
  }
  return args[key];
}

export const imageEndpoint = "/v1/images/generations";
export async function generateImage(model, prompt, signal, { size, seed } = {}) {
  const request = { model, prompt, response_format: "b64_json" };
  if (size) request.size = size;
  if (seed !== undefined) request.seed = seed;
  const result = await (await post(imageEndpoint, request, signal)).json();
  const encoded = result.data?.[0]?.b64_json;
  if (typeof encoded !== "string" || !encoded) throw new Error("No PNG returned");
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < bytes.length; ++i) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: "image/png" });
}

export function fileURL(file, signal) {
  return new Promise((resolve, reject) => {
    signal?.throwIfAborted();
    const reader = new FileReader();
    const abort = () => reader.abort();
    signal?.addEventListener("abort", abort, { once: true });
    reader.onabort = () => reject(signal.reason);
    reader.onloadend = () => signal?.removeEventListener("abort", abort);
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
}

export async function messages(entries, signal) {
  return Promise.all(
    entries
      .filter((entry) => entry.message && !entry.partial)
      .map(async ({ message, toolUpdate }) => {
        if (!Array.isArray(message.content))
          return toolUpdate
            ? {
                ...message,
                content: message.content ? toolUpdate + "\n\n" + message.content : toolUpdate,
              }
            : message;
        const parts = await Promise.all(
          message.content.map(async (part) => {
            if (part.type === "image")
              return { type: "image_url", image_url: { url: await fileURL(part.file, signal) } };
            if (part.type === "document") return { type: "text", text: documentText(part) };
            return part;
          }),
        );
        if (toolUpdate) parts.unshift({ type: "text", text: toolUpdate });
        const content = parts.every((part) => part.type === "text")
          ? parts.map((part) => part.text).join("\n\n")
          : parts;
        return { ...message, content };
      }),
  );
}
