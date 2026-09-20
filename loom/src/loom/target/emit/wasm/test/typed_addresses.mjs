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

// Compare the entire sentinel-filled buffer, not just the returned load. This
// catches truncation before a stride product and dropped/doubled static terms.
function exchange(name, pointer, arguments_, address) {
  const expected = new Uint8Array(288).fill(0xA7);
  const view = new DataView(expected.buffer);
  view.setInt32(address, -987654321, true);
  expected.forEach((value, i) => exports.write_byte(pointer + i, value));
  assert.equal(exports[name](pointer, ...arguments_, 123456789), -987654321, name);
  view.setInt32(address, 123456789, true);
  const actual = Uint8Array.from({length: expected.length}, (_, i) => exports.read_byte(pointer + i));
  assert.deepEqual(actual, expected, `${name}: ${arguments_}`);
}

for (const pointer of [0, 8, 64, 2048]) {
  for (const origin of [0, 8, 128]) {
    for (let position = 0; position < 8; ++position) {
      for (const name of ['exchange_wide_origin', 'exchange_prefix_before_cast']) {
        exchange(name, pointer, [BigInt(origin), position], origin + 16 + position * 4);
      }
    }
  }
  for (const origin of [64, 72, 128]) {
    for (let position = -8; position < 0; ++position) {
      exchange('exchange_wide_signed_coordinate', pointer,
               [origin, BigInt(position)], origin + 16 + (position + 8) * 4);
    }
  }
  // Each original term is wider than the address carrier. Their difference
  // fits; converting each term modulo 2^32 must preserve that difference.
  for (const left of [1n << 32n, (1n << 32n) + 8n]) {
    for (const right of [(1n << 32n) - 8n, 1n << 32n]) {
      for (let position = 0; position < 8; ++position) {
        exchange('exchange_cancelled_high_bits', pointer, [left, right, position],
                 Number(left - right) + 16 + position * 4);
      }
    }
  }
  for (const columns of [4, 7, 16]) {
    for (let row = 0; row < 4; ++row) {
      for (let column = 0; column < 4; ++column) {
        exchange('exchange_wide_stride', pointer, [BigInt(columns), row, column],
                 24 + (row * columns + column) * 4);
      }
    }
  }
}

for (const value of [-2147483648n, -2147483647n, -1n, 0n, 1n, 2147483647n]) {
  assert.equal(exports.narrow_index(value), Number(value));
}
for (const value of [0n, 1n, 2147483647n, 2147483648n, 4294967295n]) {
  assert.equal(exports.narrow_offset(value), Number(BigInt.asIntN(32, value)));
}
