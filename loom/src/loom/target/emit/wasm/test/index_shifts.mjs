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

// The Wasm32 index carrier and JavaScript's Wasm i32 ABI are signed. Keep the
// reference arithmetic in BigInt so host Number bitwise coercions cannot hide
// an incorrect logical/arithmetic shift or a lost high bit.
function reference(value, count) {
  const signed = BigInt(value);
  const unsigned = BigInt.asUintN(32, signed);
  const amount = BigInt(count);
  return [signed << amount, unsigned >> amount, signed >> amount].map(
      result => Number(BigInt.asIntN(32, result)));
}

const values = new Set([0, 1, -1, 0x7fffffff, -0x80000000,
                        0x55555555, -0x55555556]);
for (let bit = 0n; bit < 32n; ++bit) {
  const power = 1n << bit;
  for (const value of [power - 1n, power, power + 1n, -power - 1n,
                       -power, -power + 1n]) {
    values.add(Number(BigInt.asIntN(32, value)));
  }
}
let seed = 0x12345678n;
for (let sample = 0; sample < 256; ++sample) {
  seed = BigInt.asUintN(32, seed * 1664525n + 1013904223n);
  values.add(Number(BigInt.asIntN(32, seed)));
}

for (const value of values) {
  for (let count = 0; count < 32; ++count) {
    const expected = reference(value, count);
    assert.deepEqual(exports.index_shift_dynamic(value, count), expected.slice(0, 2),
                     `dynamic value ${value}, count ${count}`);
    assert.equal(exports.index_shift_arithmetic(value, count), expected[2],
                 `arithmetic value ${value}, count ${count}`);
    assert.deepEqual(exports.index_shift_bounded(value, count), expected,
                     `bounded value ${value}, count ${count}`);
  }
  assert.equal(exports.index_shift_constant_four(value), reference(value, 4)[0],
               `constant count 4, value ${value}`);
  assert.deepEqual(exports.index_shift_constant_boundaries(value),
                   [...reference(value, 0), ...reference(value, 31)],
                   `constant count boundaries, value ${value}`);
}
