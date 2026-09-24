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

// Evaluate the BF16 number from its sign, exponent and fraction. Every finite
// value is exact in binary32; no reinterpretation duplicates the lowering.
function reference(payload) {
  const exponent = (payload >>> 7) & 255;
  const fraction = payload & 127;
  const magnitude = exponent === 255 ? (fraction ? NaN : Infinity) :
      exponent === 0 ? fraction * 2 ** -133 :
      (1 + fraction / 128) * 2 ** (exponent - 127);
  return payload & 0x8000 ? -magnitude : magnitude;
}

function writeWord(address, payload) {
  exports.write_byte(address, 0, payload & 255);
  exports.write_byte(address, 1, payload >>> 8);
}

function readWord(address) {
  return exports.read_byte(address, 0) | exports.read_byte(address, 1) << 8;
}

for (let payload = 0; payload < 65536; ++payload) {
  const expected = reference(payload);
  assert.equal(exports.widen_bfloat(payload), expected, `value ${payload}`);
  // The logical BF16 value owns only the low half of its integer carrier.
  assert.equal(exports.widen_bfloat(payload | 0xA5A50000), expected,
               `carrier ${payload}`);

  const origin = payload % 129;
  const position = payload % 16;
  const address = 4096 + origin + position * 2;
  exports.write_byte(address - 1, 0, 0xA7);
  exports.write_byte(address, 2, 0x5D);
  writeWord(address, payload);
  assert.equal(exports.load_bfloat(address), expected, `load ${payload}`);
  assert.equal(exports.load_bfloat_at(4096, origin, position), expected,
               `indexed load ${payload}`);
  assert.equal(readWord(address), payload, `input ${payload}`);

  const replacement = (payload * 37 + 19) & 65535;
  assert.deepEqual(exports.replace_bfloat(address, replacement),
                   [expected, reference(replacement)], `replace ${payload}`);
  assert.equal(readWord(address), replacement, `stored bits ${payload}`);
  assert.equal(exports.read_byte(address - 1, 0), 0xA7);
  assert.equal(exports.read_byte(address, 2), 0x5D);

  writeWord(8192, payload);
  writeWord(8194, replacement);
  assert.equal(exports.bfloat_pair_sum(8192),
               Math.fround(expected + reference(replacement)), `sum ${payload}`);
}

// A two-byte BF16 access at the linear-memory end must not read padding.
for (const payload of [0, 0x8000, 1, 0x7F7F, 0x7F80, 0x7F81, 0xFFFF]) {
  writeWord(65534, payload);
  assert.equal(exports.load_bfloat(65534), reference(payload));
  assert.deepEqual(exports.replace_bfloat(65534, payload ^ 0x8000),
                   [reference(payload), reference(payload ^ 0x8000)]);
}
assert.throws(() => exports.load_bfloat(65535), WebAssembly.RuntimeError);
