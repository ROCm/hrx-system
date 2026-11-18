// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real WebGPU integration with an inline HAL proactor. Each C reactor export
// returns before waiting; only native promises and semaphore wakes drive it.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {WASI} from 'node:wasi';

const require = createRequire(import.meta.url);
const packageRoot = process.env.IREE_WEBGPU_PACKAGE_ROOT;
const packagePath = require.resolve('webgpu', packageRoot ? {paths: [packageRoot]} : undefined);
const {create, globals} = await import(pathToFileURL(packagePath));
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
assert(adapter, 'A native WebGPU adapter is required');
const device = await adapter.requestDevice();
const nativeErrors = [];
device.addEventListener('uncapturederror', (event) => nativeErrors.push(event.error));
const context = {memory: null, exports: null, inline: true, gpu,
  fileObjects: new Map()};
const wasi = new WASI({version: 'preview1', args: [], env: process.env});
const imports = wasi.getImportObject();
for (const [name, values] of Object.entries(createWasmImports(context))) {
  imports[name] = {...imports[name], ...values};
}
const deviceHandle = context.insertWebGPUHandle(device);
const wasmPath = resolve(dirname(fileURLToPath(import.meta.url)), __IREE_WASM_BINARY);
const {instance} = await WebAssembly.instantiate(readFileSync(wasmPath), imports);
context.memory = instance.exports.memory;
const api = instance.exports;
wasi.initialize(instance);

let waiting = null;
function checkWait() {
  if (!waiting) return;
  const value = api.query(waiting.independent);
  if (value !== 0) {
    const {resolve} = waiting;
    waiting = null;
    resolve(value);
  }
}
context.exports = {proactor_drain() {
  assert.equal(api.proactor_drain(), 0, 'proactor completion');
  checkWait();
}};
function waitFor(independent) {
  return new Promise((resolve) => {
    assert.equal(waiting, null);
    waiting = {independent, resolve};
    checkWait();
  });
}

let result = 1;
try {
  assert.equal(api.initialize(deviceHandle), 0, 'HAL initialization');
  for (const form of [0, 1, 2, 3, 4, 5, 6, 7]) {
    const length = api.file_length(form);
    for (const count of form & 4 ? [form & 1 ? 2 : 1] : [0, 1, 65]) {
      console.log(`finite file form=${form}, count=${count}`);
      const source = new Uint8Array(length);
      for (let i = 0; i < length; ++i) source[i] = (i * 17 + 3) & 255;
      const target = new Uint8Array(length).fill(0xCD);
      context.fileObjects.set(1, source.buffer);
      context.fileObjects.set(2, target.buffer);
      assert.equal(api.begin(form, count), 0, 'enqueue captures');
      assert.equal(await waitFor(1), 1, 'independent work before producer');
      assert.equal(api.query(0), 0, 'dependent file group stays pending');
      assert.equal(api.open_gate(), 0, 'release metadata producer');
      assert.equal(await waitFor(0), 1, 'terminal completion');
      assert.equal(api.finish(), 0, 'exact bytes and retained captures');
      assert.deepEqual(target, new Uint8Array(context.memory.buffer,
          api.expected_file_pointer(), length), 'JS file exact bytes');
      context.fileObjects.clear();
    }
  }
  // The third live metadata page is malformed, after valid native work has
  // already been accepted. Failure must retire that work and every capture.
  const length = api.file_length(2);
  context.fileObjects.set(1, Uint8Array.from({length}, (_, i) => (i * 17 + 3) & 255).buffer);
  context.fileObjects.set(2, new Uint8Array(length).fill(0xCD).buffer);
  assert.equal(api.begin(2, 66), 0);
  assert.equal(await waitFor(1), 1);
  assert.equal(api.open_gate(), 0);
  assert.equal(await waitFor(0), -11, 'malformed binding reports OUT_OF_RANGE');
  assert.equal(api.finish(), 0, 'failure retirement');
  api.shutdown();
  assert.deepEqual(nativeErrors, [], 'no uncaptured WebGPU validation errors');
  console.log('Finite WebGPU file groups passed');
  result = 0;
} finally {
  device.destroy();
}
// Dawn holds the event loop even after device destruction.
process.exit(result);
