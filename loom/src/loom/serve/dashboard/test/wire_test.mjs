// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from "node:assert/strict";
import test from "node:test";
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
