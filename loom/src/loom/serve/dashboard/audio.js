// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { checked, endpoint, post } from "./wire.js";

export function microphone(signal) {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
    throw new Error("Microphone capture requires localhost or HTTPS and Web Audio");
  }
  signal.throwIfAborted();
  const requested = navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    requested
      .then((stream) => {
        if (signal.aborted) stream.getTracks().forEach((track) => track.stop());
        else resolve(stream);
      }, reject)
      .finally(() => signal.removeEventListener("abort", abort));
  });
}
// Finite uploads carry the actual capture rate. In voice chat keep a short
// rolling lead-in, then trim the silence that ends an utterance before upload.
export async function record(finish, signal, onFrames, callAudio) {
  const context = callAudio?.context || new AudioContext();
  let stream, node, source, rejectCapture;
  const abort = () => rejectCapture?.(signal.reason);
  try {
    await context.resume();
    await context.audioWorklet.addModule(new URL("./capture.js", import.meta.url));
    signal.throwIfAborted();
    stream = callAudio?.stream || (await microphone(signal));
    signal.throwIfAborted();
    if (stream.getAudioTracks()[0]?.readyState !== "live")
      throw new Error("Microphone disconnected");
    node = new AudioWorkletNode(context, "loom-capture", {
      channelCount: 1,
      channelCountMode: "explicit",
    });
    source = context.createMediaStreamSource(stream);
    const pieces = [];
    let frames = 0,
      lastSpeech = 0,
      speechEnd = 0,
      ended = false;
    const flushed = new Promise((resolve, reject) => {
      rejectCapture = reject;
      node.onprocessorerror = () => reject(new Error("Microphone worklet stopped"));
      stream.getAudioTracks()[0].onended = () => reject(new Error("Microphone disconnected"));
      node.port.onmessage = ({ data }) => {
        if (data === "finished") return resolve();
        if (ended) return;
        pieces.push(data);
        frames += data.byteLength / 2;
        if (callAudio) {
          const samples = new Int16Array(data);
          let energy = 0;
          for (const sample of samples) energy += sample * sample;
          if (energy > samples.length * 400 * 400) {
            lastSpeech = frames;
            speechEnd = pieces.length;
          } else if (!speechEnd) {
            while (
              pieces.length > 1 &&
              frames - pieces[0].byteLength / 2 >= context.sampleRate * 0.06
            )
              frames -= pieces.shift().byteLength / 2;
            return;
          } else if (frames - lastSpeech >= context.sampleRate * 0.65) {
            pieces.length = speechEnd;
            ended = true;
            resolve();
            return;
          }
        }
        if (frames > context.sampleRate * 120)
          return reject(new Error("Dictation exceeded two minutes"));
        onFrames(frames);
      };
    });
    signal.addEventListener("abort", abort, { once: true });
    source.connect(node);
    // The worklet never copies input to output. Connecting it keeps it pulling.
    node.connect(context.destination);
    const port = node.port;
    finish?.then(() => port.postMessage("finish"));
    await flushed;
    if (!frames) throw new Error("Recording is empty");
    return new Blob(pieces, { type: "audio/pcm;rate=" + context.sampleRate + ";channels=1" });
  } finally {
    signal.removeEventListener("abort", abort);
    if (!callAudio) stream?.getTracks().forEach((track) => track.stop());
    else if (stream) stream.getAudioTracks()[0].onended = null;
    source?.disconnect();
    node?.disconnect();
    node?.port.close();
    if (!callAudio) await context.close();
  }
}

export async function transcribe(model, audio, signal) {
  const form = new FormData();
  form.set("model", model);
  form.set("file", audio, audio.name || "recording.pcm");
  const response = await checked(
    await fetch(endpoint("/v1/audio/transcriptions"), { method: "POST", body: form, signal }),
  );
  const { text } = await response.json();
  if (typeof text !== "string") throw new Error("Missing transcript");
  return text;
}

export async function speak(model, input, values, signal, onFrames, callContext) {
  const voice = values.voice?.trim();
  if (!voice) throw new Error("Enter a voice ID in the Voice settings");
  const context = callContext || new AudioContext();
  try {
    await context.resume();
    const request = { model, input, voice, response_format: "pcm" };
    if (values.seed !== undefined) request.seed = values.seed;
    if (values.speed) request.speed = Number(values.speed);
    const response = await post("/v1/audio/speech", request, signal);
    return await play(response, context, signal, onFrames);
  } finally {
    if (!callContext) await context.close();
  }
}

// Fetch reads and PCM samples have different boundaries. Keep at most a
// quarter second of scheduled audio plus the current network chunk.
export async function play(response, context, signal, onRendered) {
  const type = response.headers.get("Content-Type") || "";
  if (!/^audio\/pcm\s*;\s*rate=24000\s*;\s*channels=1$/i.test(type))
    throw new Error("Unexpected speech format: " + type);
  const reader = response.body.getReader();
  const pending = new Map();
  let next = context.currentTime,
    tail = -1,
    rendered = 0;
  function stop() {
    for (const source of pending.keys()) source.stop();
  }
  signal.addEventListener("abort", stop, { once: true });
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let bytes = value;
      if (tail >= 0) {
        bytes = new Uint8Array(value.length + 1);
        bytes[0] = tail;
        bytes.set(value, 1);
      }
      tail = bytes.length % 2 ? bytes[bytes.length - 1] : -1;
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const frames = Math.floor(bytes.length / 2);
      for (let i = 0; i < frames; i += 2048) {
        while (next - context.currentTime > 0.25 && pending.size)
          await pending.values().next().value;
        signal.throwIfAborted();
        const count = Math.min(frames - i, 2048);
        const buffer = context.createBuffer(1, count, 24000);
        const channel = buffer.getChannelData(0);
        for (let j = 0; j < count; ++j) channel[j] = view.getInt16(2 * (i + j), true) / 32768;
        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(context.destination);
        pending.set(
          source,
          new Promise((resolve) => {
            source.onended = () => {
              pending.delete(source);
              source.disconnect();
              if (!signal.aborted) {
                rendered += count;
                onRendered(rendered);
              }
              resolve();
            };
          }),
        );
        next = Math.max(next, context.currentTime);
        source.start(next);
        next += count / 24000;
      }
    }
    if (tail >= 0) throw new Error("Speech ended inside a PCM sample");
    await Promise.all(pending.values());
    signal.throwIfAborted();
    return rendered;
  } finally {
    signal.removeEventListener("abort", stop);
    stop();
    reader.releaseLock();
  }
}
