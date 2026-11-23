// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { ask, field, selectChoices } from "./dialog.js";
import { byId, button, disclosure, element } from "./view.js";
import { identity, readMcp, writeMcp } from "./storage.js";

const servers = new Map(),
  workers = new Map(),
  credentials = new Map();
const trigger = byId("mcp");
let database,
  selected = [];
export async function loadMcp(db) {
  database = db;
  if (!db) return;
  const rows = await readMcp(db);
  servers.clear();
  for (const row of rows) servers.set(row.id, row);
  restoreMcp();
}
function catalogTool(server, binding) {
  const tool = server?.tools.find((t) => t.name === binding.name);
  return tool &&
    tool.description === binding.fn.description &&
    JSON.stringify(tool.inputSchema) === JSON.stringify(binding.fn.parameters)
    ? tool
    : undefined;
}
export function restoreMcp({ mcp = [] } = { mcp: selected }) {
  selected = mcp.filter((b) => catalogTool(servers.get(b.server), b));
  trigger.textContent =
    "MCP servers" + (selected.length ? " · " + selected.length + " enabled" : "…");
}
export const mcpSettings = () => (selected.length ? { mcp: selected } : {});
export function offeredMcp(values, name) {
  const binding = values?.mcp?.find((b) => b.fn.name === name);
  return (
    binding &&
    catalogTool(servers.get(binding.server), binding) &&
    selected.find(
      (b) => b.server === binding.server && b.name === binding.name && b.fn.name === name,
    )
  );
}
function disconnect(id) {
  workers.get(id)?.postMessage(null);
  workers.delete(id);
}
window.addEventListener("pagehide", () => {
  for (const id of workers.keys()) disconnect(id);
});
async function request(server, data, signal) {
  signal.throwIfAborted();
  let worker = workers.get(server.id);
  if (!worker) {
    worker = new Worker(new URL("mcp-worker.js", document.baseURI), { type: "module" });
    workers.set(server.id, worker);
  }
  const { port1, port2 } = new MessageChannel();
  let stop, failure;
  try {
    return await new Promise((resolve, reject) => {
      stop = () => {
        port1.postMessage(null);
        reject(signal.reason);
      };
      failure = () => {
        if (workers.get(server.id) === worker) workers.delete(server.id);
        worker.terminate();
        reject(Error("MCP worker failed to load"));
      };
      worker.addEventListener("error", failure, { once: true });
      signal.addEventListener("abort", stop, { once: true });
      port1.onmessage = ({ data }) =>
        data.error ? reject(Error(data.error)) : resolve(data.value);
      worker.postMessage(
        { ...data, url: server.url, legacy: server.legacy, auth: credentials.get(server.id) },
        [port2],
      );
    });
  } finally {
    signal.removeEventListener("abort", stop);
    worker.removeEventListener("error", failure);
    port1.close();
  }
}
export async function callMcp(binding, source, signal) {
  const server = database ? await readMcp(database, binding.server) : servers.get(binding.server);
  const tool = catalogTool(server, binding);
  if (!tool) throw Error("MCP tool changed or was removed. Rediscover it in Tools.");
  const args = JSON.parse(source);
  if (!args || typeof args !== "object" || Array.isArray(args))
    throw Error("MCP arguments must be an object");
  return request(server, { name: tool.name, args, fields: tool.fields }, signal);
}

export async function manageMcp() {
  await loadMcp(database);
  let draft = selected.slice(),
    server,
    scope;
  const picker = element("select"),
    url = element("input"),
    secret = element("input");
  url.type = "url";
  url.placeholder = "https://mcp.exa.ai/mcp";
  secret.type = "password";
  secret.autocomplete = "off";
  const kind = selectChoices([
    ["Bearer token", "Authorization"],
    ["API key", "x-api-key"],
  ]);
  const tools = element("div"),
    status = element("output"),
    controls = element("fieldset");
  controls.className = "mcp-controls";
  function menu(id) {
    picker.replaceChildren(
      new Option("Add server…", ""),
      ...[...servers.values()].map((s) => new Option(s.title, s.id)),
    );
    picker.value = id || "";
  }
  function show() {
    server = servers.get(picker.value) || { id: identity(), url: "", tools: [] };
    url.value = server.url;
    url.readOnly = !!picker.value;
    const auth = credentials.get(server.id) || {};
    kind.value = Object.keys(auth)[0] || "Authorization";
    secret.value = (auth[kind.value] || "").replace(/^Bearer /, "");
    forget.hidden = !picker.value;
    status.textContent = server.skipped?.join("\n") || "";
    tools.replaceChildren();
    for (const tool of server.tools) {
      const mode = selectChoices([
        ["Off", "off"],
        ["Ask before running", "ask"],
        ["Auto-run", "auto"],
      ]);
      mode.setAttribute("aria-label", tool.name + " permission");
      const choice = () =>
        draft.find((b) => b.server === server.id && b.name === tool.name && catalogTool(server, b));
      const before = choice();
      mode.value = !before ? "off" : before.auto ? "auto" : "ask";
      mode.onchange = () => {
        const old = choice();
        draft = draft.filter((b) => b.server !== server.id || b.name !== tool.name);
        if (mode.value !== "off") {
          const base = tool.name.replace(/[^\w-]/g, "_").slice(0, 64);
          const used = [
            "present_html",
            "run_javascript",
            "generate_image",
            "compact_chat",
            ...draft.map((b) => b.fn.name),
          ];
          const name =
            old?.fn.name ||
            (used.includes(base) ? base.slice(0, 50) + "_" + identity().slice(0, 12) : base);
          draft.push({
            server: server.id,
            name: tool.name,
            fn: { name, description: tool.description, parameters: tool.inputSchema },
            auto: mode.value === "auto",
          });
        }
      };
      const body = element("div", tool.description);
      body.append(element("pre", JSON.stringify(tool.inputSchema, null, 2)));
      const row = element("div", undefined, "mcp-tool");
      row.append(disclosure(tool.name, body), mode);
      tools.append(row);
    }
  }
  async function edit(remove) {
    const existing = servers.get(server.id);
    controls.disabled = true;
    const save = byId("apply");
    save.disabled = true;
    status.textContent = remove ? "" : "Discovering…";
    const controller = (scope = new AbortController());
    try {
      if (remove) {
        if (database) await writeMcp(database, server.id);
        servers.delete(server.id);
        credentials.delete(server.id);
        disconnect(server.id);
        draft = draft.filter((b) => b.server !== server.id);
        menu();
        show();
      } else {
        const address = new URL(url.value);
        if (
          !["http:", "https:"].includes(address.protocol) ||
          address.username ||
          address.password ||
          address.hash
        )
          throw Error("Enter an HTTP MCP endpoint without user information or a fragment");
        if (!existing && [...servers.values()].some((s) => s.url === address.href))
          throw Error("Choose the saved server above");
        server = { ...server, url: address.href };
        const auth = secret.value
          ? { [kind.value]: (kind.value === "Authorization" ? "Bearer " : "") + secret.value }
          : {};
        if (JSON.stringify(credentials.get(server.id) || {}) !== JSON.stringify(auth) || !existing)
          disconnect(server.id);
        credentials.set(server.id, auth);
        const catalog = await request(server, {}, controller.signal);
        controller.signal.throwIfAborted();
        const row = { ...server, ...catalog };
        if (database) await writeMcp(database, existing?.id, row);
        servers.set(row.id, row);
        menu(row.id);
        show();
        status.textContent = [row.tools.length + " tools", ...row.skipped].join(" · ");
      }
      restoreMcp();
    } catch (e) {
      if (!controller.signal.aborted) status.textContent = e.message;
      if (!existing && !servers.has(server.id)) {
        disconnect(server.id);
        credentials.delete(server.id);
      }
    } finally {
      controls.disabled = false;
      if (!controller.signal.aborted) save.disabled = false;
    }
  }
  const discover = button("Discover", () => edit(false)),
    forget = button("Forget server", () => edit(true));
  picker.onchange = show;
  const actions = element("div", undefined, "row-actions");
  actions.append(discover, forget);
  const details = disclosure("Credential · this tab", field("Token or key", secret));
  details.insertBefore(field("Authentication", kind), details.lastChild);
  controls.append(
    field("Server", picker),
    field("Server URL", url),
    details,
    actions,
    status,
    tools,
  );
  menu(servers.keys().next().value);
  show();
  return new Promise((done) => {
    ask("MCP servers", [controls], "Apply to chat", () => done(draft));
    byId("action-dialog").addEventListener(
      "close",
      () => {
        scope?.abort();
        byId("apply").disabled = false;
        done();
      },
      { once: true },
    );
  });
}
