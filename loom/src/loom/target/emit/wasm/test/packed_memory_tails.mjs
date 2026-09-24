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

const copies = [
  ['copy_byte_tail_3', 1, 9],
  ['copy_byte_tail_7', 1, 21],
  ['copy_byte_tail_17', 1, 17],
  ['copy_byte_tail_63', 1, 63],
  ['copy_word_tail_7', 2, 42],
];

// Permuting the payload by position puts every byte/word encoding in every
// lane, including all floating NaN payloads, without numeric interpretation.
for (const [name, origin, length] of copies) {
  const patterns = origin === 2 ? 65536 : 256;
  const source = new Uint8Array(origin + length);
  const expected = new Uint8Array(origin + length + 8);
  for (let payload = 0; payload < patterns; ++payload) {
    if (origin === 2) {
      const words = new DataView(source.buffer);
      for (let position = 0; position < source.length; position += 2) {
        words.setUint16(position, payload + position * 37, true);
      }
    } else {
      source.forEach((_, position) => source[position] = (payload + position * 37) & 255);
    }
    expected.fill(0xA7);
    expected.set(source.subarray(origin), 4 + origin);
    write(4096, source);
    write(8192, new Uint8Array(expected.length).fill(0xA7));
    exports[name](4096, 8196);
    assert.deepEqual(read(8192, expected.length), expected, `${name}: ${payload}`);
    assert.deepEqual(read(4096, source.length), source, `${name}: source`);
  }

  // The linear-memory boundary detects widened reads as well as writes.
  const end = 65536 - origin - length;
  write(end, source);
  exports[name](end, 12288);
  assert.deepEqual(read(12288 + origin, length), source.subarray(origin));
  exports[name](12288, end);
  assert.deepEqual(read(end + origin, length), source.subarray(origin));
  assert.throws(() => exports[name](end + origin, 12288), WebAssembly.RuntimeError);
  assert.throws(() => exports[name](12288, end + origin), WebAssembly.RuntimeError);
}

for (let payload = 0; payload < 256; ++payload) {
  const original = Uint8Array.of(payload, payload + 79, payload + 193);
  const replacement = Uint8Array.of(payload + 17, payload + 128, payload + 251);
  write(4096, original);
  write(16384, replacement);
  write(8192, new Uint8Array(8).fill(0xA7));
  exports.capture_byte_tail(4096, 8193, 16384);
  assert.deepEqual(read(8192, 8), Uint8Array.of(0xA7, ...original, ...replacement, 0xA7));
  assert.deepEqual(read(4096, 3), replacement);
}
