// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { events } from "./events.js";

// One worker owns one configured endpoint and its negotiated MCP session.
const current = "2026-07-28",
  handshake = "2025-11-25",
  client = { name: "loom-serve", version: "1" };
const prefix = "io.modelcontextprotocol/";
let url,
  auth,
  connection,
  sequence = 0;
let queue = Promise.resolve();

function encoded(value) {
  const s = String(value);
  if (/^[\x20-\x7e]*$/.test(s) && s === s.trim() && !/^=\?base64\?.*\?=$/.test(s)) return s;
  return (
    "=?base64?" +
    btoa(Array.from(new TextEncoder().encode(s), (n) => String.fromCharCode(n)).join("")) +
    "?="
  );
}

// Header annotations can only follow nested properties of primitive type.
function parameterHeaders(schema) {
  const fields = [],
    seen = new Set();
  function visit(node, path) {
    if (!node || typeof node !== "object") return;
    const key = node["x-mcp-header"];
    if (key !== undefined) {
      if (
        !path?.length ||
        typeof key !== "string" ||
        !/^[!#$%&'*+.^_\x60|~A-Za-z0-9-]+$/.test(key) ||
        seen.has(key.toLowerCase()) ||
        !["integer", "boolean", "string"].includes(node.type)
      )
        throw Error("Invalid MCP parameter header");
      seen.add(key.toLowerCase());
      fields.push([key, path, node.type]);
    }
    for (const [k, v] of Object.entries(node))
      if (k === "properties" && path)
        for (const [name, sub] of Object.entries(v)) visit(sub, [...path, name]);
      else visit(v, null);
  }
  visit(schema, []);
  return fields;
}

async function rpc(state, method, params, signal, { notify, fields } = {}) {
  const id = ++sequence,
    headers = {
      ...auth,
      Accept: "application/json, text/event-stream",
      "Content-Type": "application/json",
    };
  if (state.version) headers["MCP-Protocol-Version"] = state.version;
  if (state.session) headers["Mcp-Session-Id"] = state.session;
  if (state.version === current) {
    params._meta = {
      [prefix + "protocolVersion"]: current,
      [prefix + "clientInfo"]: client,
      [prefix + "clientCapabilities"]: {},
    };
    headers["Mcp-Method"] = method;
    if (params.name) headers["Mcp-Name"] = encoded(params.name);
    for (const [key, path, type] of fields || []) {
      const value = path.reduce(
        (parent, k) => (parent && Object.hasOwn(parent, k) ? parent[k] : undefined),
        params.arguments,
      );
      if (value === undefined) continue;
      if (type === "integer" ? !Number.isSafeInteger(value) : typeof value !== type)
        throw Error("Invalid argument · " + path.join("."));
      headers["Mcp-Param-" + key] = encoded(value);
    }
  }
  const send = (body, signal) =>
    fetch(url, {
      method: "POST",
      headers,
      credentials: "omit",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal,
      body: JSON.stringify({ jsonrpc: "2.0", ...body }),
    });
  const abort = () => {
    if (state.version !== current && method === "tools/call")
      send({ method: "notifications/cancelled", params: { requestId: id } }).catch(() => {});
  };
  signal?.addEventListener("abort", abort, { once: true });
  try {
    const response = await send({ id: notify ? undefined : id, method, params }, signal);
    if (!response.ok) {
      if (response.status === 404 && state === connection) connection = undefined;
      throw Object.assign(Error("MCP " + response.status + " " + response.statusText), {
        status: response.status,
      });
    }
    if (method === "initialize") state.session = response.headers.get("Mcp-Session-Id");
    if (notify) return;
    let message;
    if (response.headers.get("Content-Type")?.startsWith("text/event-stream")) {
      for await (const data of events(response.body)) {
        const packet = JSON.parse(data);
        if (packet.id === id && !packet.method) {
          message = packet;
          break;
        }
        if (packet.method && packet.id !== undefined && state.version !== current)
          await send(
            {
              id: packet.id,
              ...(packet.method === "ping"
                ? { result: {} }
                : { error: { code: -32601, message: "Unsupported client method" } }),
            },
            signal,
          );
      }
    } else message = await response.json();
    if (message?.id !== id) throw Error("Missing MCP reply · " + method);
    if (message.error)
      throw Object.assign(Error(message.error.message), { code: message.error.code });
    if (message.result?.resultType === "input_required")
      throw Error("This tool requires interactive input from an MCP client");
    return message.result;
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}

async function connect(legacy, signal) {
  let state = { version: current },
    info;
  if (!legacy)
    try {
      info = await rpc(state, "server/discover", {}, signal);
      if (!info.supportedVersions?.includes(current)) throw Error("Unsupported MCP protocol");
    } catch (e) {
      signal.throwIfAborted();
      // Older servers may refuse the modern Mcp-Method CORS preflight.
      if (
        !(
          e instanceof TypeError ||
          [400, 404, 405].includes(e.status) ||
          [-32600, -32601, -32014].includes(e.code)
        )
      )
        throw e;
    }
  if (!info) {
    state = {};
    info = await rpc(
      state,
      "initialize",
      { protocolVersion: handshake, clientInfo: client, capabilities: {} },
      signal,
    );
    state.version = info.protocolVersion;
    if (![handshake, "2025-06-18", "2025-03-26"].includes(state.version))
      throw Error("Unsupported MCP version · " + state.version);
    await rpc(state, "notifications/initialized", {}, signal, { notify: true });
  }
  if (!info.capabilities?.tools) throw Error("This MCP server offers no tools");
  const identity = info.serverInfo || info._meta?.[prefix + "serverInfo"];
  state.title = [identity?.name || new URL(url).host, identity?.version].filter(Boolean).join(" ");
  return state;
}

async function dispatch(request, signal) {
  const state = (connection ??= await connect(request.legacy, signal));
  if (request.name) {
    const result = await rpc(
      state,
      "tools/call",
      { name: request.name, arguments: request.args },
      signal,
      { fields: request.fields },
    );
    const blocks = (result.content || []).map((b) =>
      b.type === "text"
        ? b.text
        : b.type === "resource_link"
          ? [b.name, b.uri].join(" · ")
          : b.type === "resource" && typeof b.resource?.text === "string"
            ? b.resource.text
            : "[MCP " + b.type + " omitted from text output]",
    );
    if (result.structuredContent !== undefined) {
      const structured = JSON.stringify(result.structuredContent);
      if (!blocks.includes(structured)) blocks.push(structured);
    }
    return { text: blocks.join("\n\n"), failed: !!result.isError };
  }
  const tools = [],
    names = new Set(),
    cursors = new Set(),
    skipped = [];
  let cursor;
  do {
    const page = await rpc(state, "tools/list", cursor ? { cursor } : {}, signal);
    for (const { name, description, inputSchema } of page.tools) {
      if (!name || typeof name !== "string" || names.has(name) || inputSchema?.type !== "object")
        throw Error("Invalid MCP tool definition");
      names.add(name);
      try {
        tools.push({
          name,
          description,
          inputSchema,
          fields: state.version === current ? parameterHeaders(inputSchema) : [],
        });
      } catch (e) {
        skipped.push(name + " · " + e.message);
      }
    }
    cursor = page.nextCursor;
    if (cursor && cursors.has(cursor)) throw Error("Repeated MCP catalog cursor");
    cursors.add(cursor);
  } while (cursor);
  return { title: state.title, legacy: state.version !== current, tools, skipped };
}

self.onmessage = ({ data, ports: [port] }) => {
  if (!port) {
    // Termination is optional for handshake-era servers. Keep it independent
    // of the worker lifetime and never leave an external request blocking close.
    if (connection?.session)
      fetch(url, {
        method: "DELETE",
        keepalive: true,
        credentials: "omit",
        redirect: "error",
        referrerPolicy: "no-referrer",
        headers: {
          ...auth,
          "Mcp-Session-Id": connection.session,
          "MCP-Protocol-Version": connection.version,
        },
      }).catch(() => {});
    self.close();
    return;
  }
  const scope = new AbortController();
  port.onmessage = () => scope.abort();
  queue = queue.then(async () => {
    try {
      scope.signal.throwIfAborted();
      url = data.url;
      auth = data.auth;
      port.postMessage({ value: await dispatch(data, scope.signal) });
    } catch (e) {
      port.postMessage({ error: e.message });
    } finally {
      port.close();
    }
  });
};
