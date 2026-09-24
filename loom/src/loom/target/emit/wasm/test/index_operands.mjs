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

for (const input of [0, 1, 511, 512, 1023]) {
  assert.equal(exports.high_bit_offset_add(input), 2147483648n + BigInt(input));
}

// Address results cross the Wasm i32 ABI as signed JavaScript numbers, while
// widening inside the compiled program must retain the source domain's value.
assert.deepEqual(exports.address_constant_boundaries(),
                 [-2147483648, 2147483647, 1023, 1024, -2147483648, -1]);
assert.deepEqual(exports.offset_from_constant_payloads(), [-2147483648, -1]);

// BigInt keeps the full product before reduction to the target's signed i32
// carrier, including products wider than JavaScript's exact Number range.
const coordinates = [-2147483648, -2147483647, -65537, -1, 0, 1, 65537, 2147483647];
for (const a of coordinates) {
  for (const b of coordinates) {
    const coordinate = Number(BigInt.asIntN(32, BigInt(a) * 8n + BigInt(b)));
    assert.equal(exports.logical_index_mul_add(a, b), coordinate);
    for (const c of coordinates) {
      const sum = Number(BigInt.asIntN(32, BigInt(a) * BigInt(b) + BigInt(c)));
      assert.deepEqual(exports.logical_index_madd(a, b, c), [sum, a, b, c]);
    }
  }
}
