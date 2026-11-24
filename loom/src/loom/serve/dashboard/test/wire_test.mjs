// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from "node:assert/strict";
import test from "node:test";
import { editable, isUserTurn } from "../content.js";
import { events } from "../events.js";
import { appendCalls, chatChunks, messages, offeredTools, toolArgument, toolChange } from "../wire.js";

async function collect(stream) {
  const items = [];
  for await (const item of stream) items.push(item);
  return items;
}

test("SSE preserves split Unicode, CRLF, and multiline fields", async () => {
  const bytes = new TextEncoder().encode(": idle\r\ndata: café\r\ndata: 水\r\n\r\ndata\n\ndata: incomplete");
  const stream = new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    },
  });
  assert.deepEqual(await collect(events(stream)), ["café\n水", ""]);
});

test("a chat stream requires its terminal event even after a valid delta", async () => {
  const response = (text) => new Response(text, { headers: { "Content-Type": "text/event-stream" } });
  const text = 'data: {"choices":[]}\n\n';
  assert.deepEqual(await collect(chatChunks(response(text + "data: [DONE]\n\n"))), [{ choices: [] }]);
  await assert.rejects(collect(chatChunks(response(text))), /final chat event/);
});

test("fragmented tool arguments are complete before the call runs", () => {
  const calls = [];
  appendCalls(calls, [{ index: 0, id: "call-1", function: { name: "present_", arguments: '{"html":' } }]);
  appendCalls(calls, [{ index: 0, function: { name: "html", arguments: '"<p>hi</p>"}' } }]);
  assert.equal(calls[0].function.name, "present_html");
  assert.equal(toolArgument(calls[0], "html"), "<p>hi</p>");
  assert.throws(() => appendCalls(calls, [{ index: 2 }]), /tool call index/);
  calls[0].function.arguments = '{"html":"one","html":"two"}';
  assert.throws(() => toolArgument(calls[0], "html"), /one nonempty string/);
});

test("history sends completed messages and carries tool availability on the next input", async () => {
  const update = toolChange({ htmlTool: true }, offeredTools({ codeTool: true }));
  assert.match(update, /Enabled: run_javascript/);
  assert.match(update, /Disabled: present_html/);
  const text = { role: "user", content: [{ type: "document", name: "main.c", text: "int main();" }] };
  const history = await messages([
    { message: { role: "assistant", content: "incomplete" }, partial: true },
    { message: text, toolUpdate: update },
  ]);
  assert.deepEqual(history, [{ role: "user", content: update + "\n\n[File: main.c]\nint main();" }]);
});

test("a compacted history starts with the latest summary and its prompt", async () => {
  const entry = (role, content) => ({ message: { role, content } });
  const first = entry("user", "Look up the specification");
  const original = entry("tool", "Full search output");
  original.message.tool_call_id = "search";
  const call = entry("assistant", "");
  call.message.tool_calls = [
    { type: "function", id: "search", function: { name: "search", arguments: "{}" } },
  ];
  const prompt = entry("user", "Summarize");
  const summary = { ...entry("assistant", "First handoff"), compaction: true };
  const latest = { ...entry("assistant", "Second handoff"), compaction: true };
  const follow = entry("user", "Continue");
  const entries = [first, call, original, prompt, summary, follow, prompt, latest, follow];
  assert.deepEqual(
    await messages(entries.slice(0, 4)),
    [first, call, original, prompt].map((e) => e.message),
  );
  assert.deepEqual(
    await messages(entries.slice(0, 6)),
    [prompt, summary, follow].map((e) => e.message),
  );
  assert.deepEqual(
    await messages(entries),
    [prompt, latest, follow].map((e) => e.message),
  );
  assert.equal(original.message.content, "Full search output");
});

test("undo finds an authored input across tool results, summary prompts, and continuations", async () => {
  const user = (content) => ({ message: { role: "user", content } });
  const summary = { compaction: true, message: { role: "assistant", content: "Handoff" } };
  const handoff = user("Summarize the conversation");
  const automatic = { ...user("Continue."), automatic: true };
  const call = { id: "lookup", type: "function", function: { name: "search", arguments: "{}" } };
  const entries = [
    user("Search for this"),
    { message: { role: "assistant", content: "", tool_calls: [call] } },
    { message: { role: "tool", tool_call_id: "lookup", content: "Search result" } },
    { ...user(""), toolUpdate: "Tool availability changed" },
    handoff,
    summary,
    automatic,
  ];
  assert.equal(entries.findLastIndex(isUserTurn), 0);
  assert.deepEqual(await messages(entries), [handoff.message, summary.message, automatic.message]);
  const typed = user("Continue.");
  entries.push(typed, { message: { role: "assistant", content: "Partial" }, partial: true });
  assert.equal(entries.findLast(isUserTurn), typed);
  assert.equal([handoff, summary, automatic].findLastIndex(isUserTurn), -1);
});

test("editing restores the composer text and attachments for chat and media inputs", () => {
  const document = { type: "document", name: "input.txt", text: "Contents" };
  const entry = { message: { role: "user", content: [{ type: "text", text: "<task>" }, document] } };
  const settings = { template: "<{{input}}>" };
  assert.deepEqual(editable(entry, settings), { text: "task", files: [document] });
  assert.deepEqual(editable({ ...entry, draft: "  task\n" }, settings), {
    text: "  task\n", files: [document],
  });
  const fileOnly = { message: { role: "user", content: [document] } };
  assert.equal([fileOnly].findLastIndex(isUserTurn), 0);
  assert.deepEqual(editable(fileOnly, {}), { text: "", files: [document] });
  const media = { role: "user", text: "<task>" };
  assert.equal([media].findLastIndex(isUserTurn), 0);
  assert.deepEqual(editable(media, settings), { text: "<task>", files: [] });
});

test("the opt-in compaction tool carries the requested summary focus", () => {
  assert.deepEqual(offeredTools({}), []);
  const tools = offeredTools({ compactTool: true, compactFocus: "Keep source IDs" });
  assert.equal(tools.length, 1);
  assert.equal(tools[0].function.name, "compact_chat");
  assert.match(tools[0].function.description, /Keep source IDs/);
  assert.match(toolChange({}, tools), /Enabled: compact_chat/);
  const call = { function: { name: "compact_chat", arguments: '{"summary":"Keep source S-23"}' } };
  assert.equal(toolArgument(call, "summary"), "Keep source S-23");
});
