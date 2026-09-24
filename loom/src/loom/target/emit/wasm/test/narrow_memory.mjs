// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const binary = readFileSync(process.argv[2]);
assert.ok(WebAssembly.validate(binary));
const {instance: {exports}} = await WebAssembly.instantiate(binary);
const write = (base, bytes) => bytes.forEach((value, i) => exports.write_byte(base, i, value));
const read = (base, length) => Uint8Array.from({length}, (_, i) => exports.read_byte(base, i));

// Raw buffer-byte operations initialize and inspect memory independently of the
// view-memory rules under test. Compare every byte, including input and guards.
function copy(name, arguments_, byte, word, sourceBase = 256, destinationBase = 1024) {
  const source = Uint8Array.from({length: 224}, (_, i) => (i * 37 + word) & 255);
  const expected = new Uint8Array(source.length).fill(0xA7);
  write(sourceBase, source);
  write(destinationBase, expected);
  exports[name](sourceBase, destinationBase, ...arguments_);
  expected[byte] = source[byte];
  expected.set(source.subarray(word, word + 2), word);
  assert.deepEqual(read(destinationBase, expected.length), expected, name);
  assert.deepEqual(read(sourceBase, source.length), source, name);
}

for (const sourceBase of [0, 1, 255]) {
  for (const destinationBase of [1024, 1025, 2047]) {
    copy('copy_narrow_static', [], 3, 5, sourceBase, destinationBase);
    for (let position = 0; position < 16; ++position) {
      copy('copy_narrow_index', [position], 1 + position, 33 + position * 2,
           sourceBase, destinationBase);
    }
  }
}
for (const base of [0, 1, 31, 128]) {
  for (let position = 0; position < 16; ++position) {
    copy('copy_narrow_origin', [base, position], base + position, base + 33 + position * 2);
  }
}
for (const columns of [1, 3, 7, 16]) {
  for (let row = 0; row < 4; ++row) {
    for (let column = 0; column < columns; ++column) {
      const position = row * columns + column;
      copy('copy_narrow_stride', [columns, row, column], 1 + position, 65 + position * 2);
    }
  }
}

// Narrow parameters occupy i32 ABI carriers. Stores must ignore their upper
// bits, while unsigned consumers must preserve high-bit-set narrow payloads.
for (const value of [0, 1, 127, 128, 255, 256, 32767, 32768, 65535, -1, 0x89ABCDEF]) {
  const expected = new Uint8Array(32).fill(0xA7);
  const view = new DataView(expected.buffer);
  expected[7] = 0xE1;
  view.setUint16(8, 0xFEDC, true);
  write(4096, expected);
  assert.deepEqual(exports.exchange_narrow(4103, value, value), [0xE1, 0xFEDC]);
  expected[7] = value;
  view.setUint16(8, value, true);
  assert.deepEqual(read(4096, expected.length), expected);
}

// Put the final halfword exactly at the linear-memory end. A widened load or
// store traps here, even if an ordinary guard comparison could miss overreads.
write(65528, Uint8Array.from([0xA7, 0xA7, 0xA7, 0xE1, 0xA7, 0xA7, 0xDC, 0xFE]));
exports.copy_narrow_static(65529, 8192);
assert.equal(exports.read_byte(8192, 3), 0xA7);
assert.deepEqual(read(8197, 2), Uint8Array.of(0xDC, 0xFE));
exports.copy_narrow_static(8192, 65529);
assert.deepEqual(read(65534, 2), Uint8Array.of(0xDC, 0xFE));
assert.throws(() => exports.copy_narrow_static(65530, 8192), WebAssembly.RuntimeError);
assert.throws(() => exports.copy_narrow_static(8192, 65530), WebAssembly.RuntimeError);

// The original repeated volatile-byte witness remains observable at the edge.
exports.write_byte(65535, 0, 0xD7);
exports.observe_byte(65535, 12288);
assert.deepEqual(read(12288, 2), Uint8Array.of(0xD7, 0xD7));

// Exercise the shared scalar addressing corpus through the default pipeline.
// The fixed-rank dynamic coordinate becomes index.madd before source lowering.
function copyScalar(name, arguments_, inputOffset, outputOffset, width = 4) {
  const length = Math.max(inputOffset, outputOffset) + width + 16;
  const source = Uint8Array.from({length}, (_, i) => (i * 37 + 13) & 255);
  const expected = new Uint8Array(length).fill(0xA7);
  const sourceView = new DataView(source.buffer);
  const value = width === 4 ? sourceView.getInt32(inputOffset, true)
                           : sourceView.getUint16(inputOffset, true);
  write(0, source);
  write(8192, expected);
  assert.equal(exports[name](0, 8192, ...arguments_), value, name);
  expected.set(source.subarray(inputOffset, inputOffset + width), outputOffset);
  assert.deepEqual(read(8192, length), expected, name);
  assert.deepEqual(read(0, length), source, name);
}

copyScalar('scalar_static_rank2_i32', [], 8 + (1 * 4 + 2) * 4, 12 + 3 * 4);
copyScalar('scalar_static_rank3_i16', [], 8 + ((1 * 3 + 2) * 4 + 3) * 2,
           12 + (1 * 4 + 2) * 2, 2);
for (let element = 0; element < 8; ++element) {
  copyScalar('scalar_dynamic_index_i32', [element], element * 4, element * 4);
}
for (const base of [0, 4, 128, 4056]) {
  copyScalar('scalar_dynamic_view_base_i32', [base], base + 8, 0);
  copyScalar('scalar_materialized_view_base_suffix_i32', [base], base + 12, base + 16);
  for (let element = 0; element < 4; ++element) {
    copyScalar('scalar_subview_dynamic_index_i32', [base, element],
               base + (1 + element) * 4, base + (1 + element) * 4);
  }
  for (let row = 0; row < 4; ++row) {
    for (let column = 0; column < 8; ++column) {
      const offset = base + (row * 8 + column) * 4;
      copyScalar('scalar_mixed_rank2_i32', [base, row, column], offset, offset);
    }
  }
}
for (const columns of [1, 3, 7, 8]) {
  for (let row = 0; row < 4; ++row) {
    for (let column = 0; column < columns; ++column) {
      const offset = (row * columns + column) * 4;
      copyScalar('scalar_dynamic_stride_i32', [columns, row, column], offset, offset);
    }
  }
}
