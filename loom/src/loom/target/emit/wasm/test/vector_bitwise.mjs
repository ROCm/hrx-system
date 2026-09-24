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
const operations = [(a, b) => a & b, (a, b) => a | b, (a, b) => a ^ b];

// One-hot and complementary patterns cross every byte and lane boundary. The
// byte oracle checks all 128 bits independently of the source element width.
for (let bit = 0; bit < 128; ++bit) {
  for (let variant = 0; variant < 4; ++variant) {
    const inputs = [0, 1].map(input => {
      const bytes = new Uint8Array(48).fill(0x73 + input);
      const invert = (variant >> input) & 1;
      bytes.fill(invert ? 0xFF : 0, 16, 32);
      bytes[16 + (bit >> 3)] ^= 1 << (bit & 7);
      write(256 + input * 64, bytes);
      return bytes;
    });
    for (const element of ['i32', 'i64']) {
      const expected = new Uint8Array(80).fill(0xA5);
      write(512, expected);
      operations.forEach((operation, index) => {
        for (let byte = 0; byte < 16; ++byte) {
          expected[16 + index * 16 + byte] = operation(
              inputs[0][16 + byte], inputs[1][16 + byte]);
        }
      });
      exports[`vector_bitwise_${element}`](272, 336, 528);
      const context = `${element}, bit=${bit}, variant=${variant}`;
      assert.deepEqual(read(512, expected.length), expected, context);
      inputs.forEach((bytes, input) => assert.deepEqual(
          read(256 + input * 64, bytes.length), bytes, context));
    }
  }
}

const truePayload = 0x513579BD;
const falsePayload = -0x1234567;
for (let lhs = 0; lhs < 16; ++lhs) {
  for (let rhs = 0; rhs < 16; ++rhs) {
    const inputs = [lhs, rhs].map((bits, input) => {
      const bytes = new Uint8Array(48).fill(0x73 + input);
      const view = new DataView(bytes.buffer);
      for (let lane = 0; lane < 4; ++lane) {
        view.setInt32(16 + lane * 4, bits & (1 << lane) ? 17 + input * 4 + lane : 0, true);
      }
      write(256 + input * 64, bytes);
      return bytes;
    });
    const expected = new Uint8Array(80).fill(0xA5);
    write(512, expected);
    const view = new DataView(expected.buffer);
    const scalars = operations.flatMap((operation, index) => {
      const bits = operation(lhs, rhs);
      for (let lane = 0; lane < 4; ++lane) {
        view.setInt32(16 + index * 16 + lane * 4,
                      bits & (1 << lane) ? truePayload : falsePayload, true);
      }
      return [bits & 1 ? -1 : 0, bits & 8 ? 1 : 0];
    });
    const context = `lhs=${lhs}, rhs=${rhs}`;
    assert.deepEqual(exports.predicate_bitwise_values(
        272, 336, 528, 0, truePayload, falsePayload), scalars, context);
    assert.deepEqual(read(512, expected.length), expected, context);
    inputs.forEach((bytes, input) => assert.deepEqual(
        read(256 + input * 64, bytes.length), bytes, context));
  }
}

// Integer bit patterns are independent of JavaScript's floating-point and NaN
// conversions. Each expected word repeats across all 128 bits of its vector.
const cases = [
  ['i32', 4, [0n, 1n, 0xFEDCBA99n, 0x80000000n, 0xFFFFFFFFn, 0x7FFFFFFFn]],
  ['i64', 8, [0n, 1n, 0x00000000FFFFFFFFn, 0xFFFFFFFF00000000n,
              0x8000000000000001n, 0x8000000000000000n, 0x7FFFFFFFFFFFFFFFn,
              0xFFFFFFFFFFFFFFFFn]],
  ['f32', 4, [0n, 0x80000000n, 0xBF8CCCCDn, 1n, 0x00800000n,
              0x7F800000n, 0xFF800000n, 0x7FC00000n, 0x3F800000n]],
  ['f64', 8, [0n, 0x8000000000000000n, 0xBFF199999999999An, 1n,
              0x0010000000000000n, 0x7FF0000000000000n, 0xFFF0000000000000n,
              0x7FF8000000000000n, 0x3FF199999999999An]],
];
for (const [element, width, words] of cases) {
  const expected = new Uint8Array(32 + words.length * 16).fill(0xA5);
  write(256, expected);
  words.forEach((word, index) => {
    for (let byte = 0; byte < 16; ++byte) {
      expected[16 + index * 16 + byte] = Number((word >> BigInt((byte % width) * 8)) & 0xFFn);
    }
  });
  exports[`vector_constants_${element}`](272);
  assert.deepEqual(read(256, expected.length), expected, element);
}

for (const condition of [0, 1]) {
  const expected = new Uint8Array(48).fill(0xA5);
  write(256, expected);
  const view = new DataView(expected.buffer);
  for (let lane = 0; lane < 4; ++lane) {
    view.setInt32(16 + lane * 4, condition ? truePayload : falsePayload, true);
  }
  assert.deepEqual(exports.vector_constants_predicate(
      272, condition, truePayload, falsePayload), condition ? [-1, 1] : [0, 0]);
  assert.deepEqual(read(256, expected.length), expected, `predicate=${condition}`);
}

// A native f64 load must preserve noncanonical NaN payloads and distinct lanes.
const float64Patterns = [
  [0x8000000000000000n, 0x0000000000000001n],
  [0x7FF8000000000123n, 0xFFF0000000000456n],
  [0x3FF199999999999An, 0xBFF199999999999An],
];
for (const words of float64Patterns) {
  const input = new Uint8Array(48).fill(0x73);
  const inputView = new DataView(input.buffer);
  words.forEach((word, lane) => inputView.setBigUint64(16 + lane * 8, word, true));
  write(256, input);
  for (const condition of [0, 1]) {
    const expected = new Uint8Array(48).fill(0xA5);
    write(512, expected);
    const outputView = new DataView(expected.buffer);
    words.forEach((word, lane) => outputView.setBigUint64(
        16 + lane * 8, condition ? 0x8000000000000000n : word, true));
    exports.vector_constants_f64_select(272, 528, condition);
    assert.deepEqual(read(512, expected.length), expected, `f64 select=${condition}`);
    assert.deepEqual(read(256, input.length), input);
  }
}
