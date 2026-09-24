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

// Decode through the mathematical significand and exponent, independently of
// the compiler's integer-bit construction. Every finite FP8 value is exact.
function decode(payload, mantissaBits, exponentBits) {
  const mantissaScale = 2 ** mantissaBits;
  const mantissa = payload % mantissaScale;
  const exponent = (payload & 127) >> mantissaBits;
  const maximumExponent = (1 << exponentBits) - 1;
  const bias = maximumExponent >> 1;
  if (exponent === maximumExponent) {
    if (mantissaBits === 3 && mantissa === 7) return NaN;
    if (mantissaBits === 2) {
      if (mantissa !== 0) return NaN;
      return payload & 128 ? -Infinity : Infinity;
    }
  }
  const value = exponent === 0
      ? (mantissa / mantissaScale) * 2 ** (1 - bias)
      : (1 + mantissa / mantissaScale) * 2 ** (exponent - bias);
  return payload & 128 ? -value : value;
}

for (let payload = 0; payload < 256; ++payload) {
  exports.write_byte(1023, 0, payload);
  write(2048, new Uint8Array(16).fill(0xA7));
  exports.extend_float8(1023, 2052);
  const bytes = read(2048, 16);
  assert.deepEqual(bytes.subarray(0, 4), new Uint8Array(4).fill(0xA7));
  assert.deepEqual(bytes.subarray(12), new Uint8Array(4).fill(0xA7));
  const values = new DataView(bytes.buffer);
  for (const [lane, mantissa, exponent] of [[0, 3, 4], [1, 2, 5]]) {
    assert.equal(values.getFloat32(4 + lane * 4, true), decode(payload, mantissa, exponent),
                 `payload=${payload}, mantissa=${mantissa}`);
  }
  assert.equal(exports.read_byte(1023, 0), payload);

  // A rotating permutation exercises every byte in every packed lane, including
  // NaN encodings. Storage must not interpret or canonicalize those payloads.
  const source = Uint8Array.from({length: 192}, (_, i) => (payload + i * 37) & 255);
  for (const [name, positions, arguments_] of [
    ['copy_float8_scalar', [3, 5], []],
    ['copy_float8_vector', Array.from({length: 32}, (_, i) => i), []],
    ['copy_float8_tail', [1, 2, 3, 4], []],
    ['copy_float8_index', [payload % 129 + payload % 16, payload % 129 + payload % 16 + 16],
     [payload % 129, payload % 16]],
  ]) {
    const expected = new Uint8Array(200).fill(0xA7);
    write(4095, source);
    write(8192, expected);
    exports[name](4095, 8195, ...arguments_);
    for (const position of positions) expected[3 + position] = source[position];
    assert.deepEqual(read(8192, expected.length), expected, `${name}: ${payload}`);
    assert.deepEqual(read(4095, source.length), source, `${name}: input`);
  }
}

// Exact spans at the end of linear memory turn widened reads into traps.
write(65532, Uint8Array.of(0x7F, 0xFF, 0x7D, 0xFD));
exports.copy_float8_tail(65531, 12288);
assert.deepEqual(read(12289, 4), Uint8Array.of(0x7F, 0xFF, 0x7D, 0xFD));
exports.copy_float8_tail(12288, 65531);
assert.deepEqual(read(65532, 4), Uint8Array.of(0x7F, 0xFF, 0x7D, 0xFD));
assert.throws(() => exports.copy_float8_tail(65532, 12288), WebAssembly.RuntimeError);
assert.throws(() => exports.copy_float8_tail(12288, 65532), WebAssembly.RuntimeError);
exports.write_byte(65535, 0, 0xFF);
exports.copy_float8_volatile(65535, 16384);
assert.deepEqual(read(16384, 2), Uint8Array.of(0xFF, 0xFF));
