// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const binary = readFileSync(process.argv[2]);
assert.ok(WebAssembly.validate(binary));
const {instance} = await WebAssembly.instantiate(binary);
const exports = instance.exports;

assert.deepEqual(exports.boolean_constants(), [0, 1]);
for (const lhs of [0, 1]) {
  assert.deepEqual(exports.boolean_unsigned_extensions(lhs), [lhs, BigInt(lhs)]);
  for (const rhs of [0, 1]) {
    assert.deepEqual(exports.boolean_algebra(lhs, rhs), [lhs & rhs, lhs | rhs, lhs ^ rhs]);
  }
}

const inputs = [-2147483648, -2147483647, -1, 0, 1, 17, 2147483647];
for (const left of inputs) {
  for (const right of inputs) {
    const lhs = Number(left !== 0);
    const rhs = Number(right !== 0);
    // Signed i1 interprets its set bit as -1; unsigned i1 interprets it as 1.
    const signedLhs = -lhs;
    const signedRhs = -rhs;
    const predicates = [
      lhs === rhs, lhs !== rhs,
      signedLhs < signedRhs, signedLhs <= signedRhs,
      signedLhs > signedRhs, signedLhs >= signedRhs,
      lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs,
    ];
    const expected = predicates.reduce((mask, value, index) => mask | (Number(value) << index), 0);
    assert.equal(exports.boolean_comparisons(left, right), expected);
  }
}
