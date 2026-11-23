// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PCM transfer uses complete little-endian samples and flushes the final tail.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.packet = new DataView(new ArrayBuffer(960));
    this.used = 0;
    this.active = true;
    this.port.onmessage = ({ data }) => {
      if (data === "finish") {
        this.active = false;
        if (this.used) this.flush();
        this.port.postMessage("finished");
      }
    };
  }
  flush() {
    const bytes =
      this.used === this.packet.byteLength
        ? this.packet.buffer
        : this.packet.buffer.slice(0, this.used);
    this.port.postMessage(bytes, [bytes]);
    this.packet = new DataView(new ArrayBuffer(960));
    this.used = 0;
  }
  process(inputs) {
    if (!this.active) return false;
    const channel = inputs[0]?.[0];
    if (channel)
      for (const input of channel) {
        const sample = Math.max(-1, Math.min(1, input));
        this.packet.setInt16(this.used, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
        this.used += 2;
        if (this.used === this.packet.byteLength) this.flush();
      }
    return true;
  }
}
registerProcessor("loom-capture", Capture);
