// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The frame owns a single worker. Removing its document retires the worker,
// including a calculation that never yields to the worker's event loop.
export async function runJavaScript(code, signal) {
  signal.throwIfAborted();
  const frame = document.createElement("iframe");
  frame.hidden = true;
  frame.title = "JavaScript tool";
  frame.sandbox = "allow-scripts";
  frame.referrerPolicy = "no-referrer";
  const { port1, port2 } = new MessageChannel();
  let stop;
  try {
    return await new Promise((resolve, reject) => {
      stop = () => resolve({ failed: true, text: "JavaScript stopped by user." });
      signal.addEventListener("abort", stop, { once: true });
      port1.onmessage = ({ data }) => resolve(data);
      port1.onmessageerror = () => reject(new Error("Unreadable JavaScript result"));
      frame.onerror = () => reject(new Error("JavaScript runner failed to load"));
      frame.onload = () => {
        // An opaque receiver requires "*" and replies only on this private port.
        try {
          frame.contentWindow.postMessage(code, "*", [port2]);
        } catch (error) {
          reject(error);
        }
      };
      frame.src = "code.html";
      document.body.append(frame);
    });
  } finally {
    signal.removeEventListener("abort", stop);
    frame.remove();
    port1.close();
    port2.close();
  }
}
