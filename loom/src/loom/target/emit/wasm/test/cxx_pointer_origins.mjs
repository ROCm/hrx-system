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
const expected = new Uint8Array(16384);
for (let i = 0; i < expected.length; ++i) {
  expected[i] = (i * 29 + (i >>> 4) * 37) & 255;
  exports.write_byte(0, 0, i, expected[i]);
}

const words = new DataView(expected.buffer);
for (const root of [0, 4, 2048, 4096]) {
  for (let origin = 16; origin <= 8188; origin += 4) {
    const accesses = [
      ['read', 0],
      ['read_zero', 0],
      ['read_next', 4],
      ['read_previous', -4],
      ['read_member', 4],
      ['read_helper', 8],
    ];
    for (const [name, displacement] of accesses) {
      assert.equal(exports[name](root, origin),
                   words.getInt32(root + origin + displacement, true),
                   `${name}: ${root}+${origin}`);
    }
    for (const index of [-2, -1, 0, 1, 2]) {
      assert.equal(exports.read_at(root, origin, index),
                   words.getInt32(root + origin + index * 4, true),
                   `read_at: ${root}+${origin}, ${index}`);
    }
  }
}

for (const root of [0, 4096, 8192]) {
  for (let index = -4; index <= 4; ++index) {
    const origin = 32;
    const address = root + origin + index * 4;
    const replacement = (root * 17 + index * 104729) | 0;
    const previous = words.getInt32(address, true);
    assert.equal(exports.exchange(root, origin, index, replacement), previous);
    words.setInt32(address, replacement, true);
  }
}

const actual = Uint8Array.from(
    {length: expected.length}, (_, i) => exports.read_byte(0, 0, i));
assert.deepEqual(actual, expected);
