// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TextDecoderStream retains split UTF-8. SSE fields can span any HTTP read.
export async function* events(body) {
  const reader = body.pipeThrough(new TextDecoderStream("utf-8", { fatal: true })).getReader();
  let pending = "",
    afterCR = false,
    fields = [];
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      let start = 0;
      for (let i = 0; i < value.length; ++i) {
        const character = value[i];
        if (afterCR) {
          afterCR = false;
          if (character === "\n") {
            start = i + 1;
            continue;
          }
        }
        if (character !== "\r" && character !== "\n") continue;
        const line = pending + value.slice(start, i);
        pending = "";
        start = i + 1;
        afterCR = character === "\r";
        if (line === "" && fields.length) {
          const data = fields.join("\n");
          fields = [];
          yield data;
        } else if (line === "data" || line.startsWith("data:")) {
          fields.push(line.slice(line[5] === " " ? 6 : 5));
        }
      }
      pending += value.slice(start);
    }
  } finally {
    try {
      await reader.cancel();
    } finally {
      reader.releaseLock();
    }
  }
}
