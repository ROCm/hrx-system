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

function checkCopy(name, source, positions, arguments_ = []) {
  const expected = new Uint8Array(source.length + 8).fill(0xA7);
  write(4096, source);
  write(8192, expected);
  exports[name](4096, 8196, ...arguments_);
  for (const position of positions) expected[4 + position] = source[position];
  assert.deepEqual(read(8192, expected.length), expected, name);
  assert.deepEqual(read(4096, source.length), source, `${name}: input`);
}

// Every 16-bit pattern passes through F16, BF16 and integer storage. The odd
// lane permutation exercises signs, exponents and NaN payloads in mixed vectors.
for (const [name, lanes] of [
  ['copy_word_scalar', 1],
  ['copy_word_vector', 8],
  ['copy_word_vector_wide', 16],
  ['copy_word_vector_large', 32],
]) {
  const source = new Uint8Array(lanes * 6);
  const words = new DataView(source.buffer);
  const positions = Array.from({length: source.length}, (_, i) => i);
  for (let payload = 0; payload < 65536; payload += lanes) {
    for (let format = 0; format < 3; ++format) {
      for (let lane = 0; lane < lanes; ++lane) {
        words.setUint16((format * lanes + lane) * 2,
                        payload + lane * 37 + format * 0x3457, true);
      }
    }
    checkCopy(name, source, positions);
  }
}

const source = Uint8Array.from({length: 224}, (_, i) => (i * 37 + 13) & 255);
checkCopy('copy_word_unaligned', source, Array.from({length: 48}, (_, i) => i + 1));
checkCopy('copy_word_tail', source, Array.from({length: 18}, (_, i) => i + 1));
for (const base of [0, 1, 31, 128]) {
  for (let position = 0; position < 8; ++position) {
    const positions = [0, 16, 32].flatMap(offset =>
      [base + offset + position * 2, base + offset + position * 2 + 1]);
    checkCopy('copy_word_index', source, positions, [base, position]);
  }
}
for (const columns of [1, 3, 7, 16]) {
  for (let row = 0; row < 4; ++row) {
    for (let column = 0; column < columns; ++column) {
      const position = 3 + (row * columns + column) * 2;
      checkCopy('copy_word_stride', source, [position, position + 1],
                [columns, row, column]);
    }
  }
}

// Exact spans at the linear-memory end catch reads or writes widened beyond
// the authored transfer, even when ordinary guard bytes would hide overreads.
const tail = Uint8Array.from({length: 18}, (_, i) => (0xFD + i * 17) & 255);
write(65518, tail);
exports.copy_word_tail(65517, 12288);
assert.deepEqual(read(12289, tail.length), tail);
exports.copy_word_tail(12288, 65517);
assert.deepEqual(read(65518, tail.length), tail);
assert.throws(() => exports.copy_word_tail(65518, 12288), WebAssembly.RuntimeError);
assert.throws(() => exports.copy_word_tail(12288, 65518), WebAssembly.RuntimeError);
for (const [name, length] of [
  ['copy_word_vector', 48],
  ['copy_word_vector_wide', 96],
  ['copy_word_vector_large', 192],
]) {
  const bytes = source.subarray(0, length);
  write(65536 - length, bytes);
  exports[name](65536 - length, 16384);
  assert.deepEqual(read(16384, length), bytes);
  exports[name](16384, 65536 - length);
  assert.deepEqual(read(65536 - length, length), bytes);
  assert.throws(() => exports[name](65538 - length, 16384), WebAssembly.RuntimeError);
  assert.throws(() => exports[name](16384, 65538 - length), WebAssembly.RuntimeError);
}
write(65534, Uint8Array.of(0x81, 0xFF));
exports.copy_word_volatile(65534, 20480);
assert.deepEqual(read(20480, 4), Uint8Array.of(0x81, 0xFF, 0x81, 0xFF));
