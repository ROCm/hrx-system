// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace arithmetic {
[[loom::symbol("arithmetic.mix")]] static unsigned mix(unsigned value);
static unsigned mix(unsigned value) { return value * value + 7u; }
}  // namespace arithmetic

[[loom::kernel, loom::symbol("library.dispatch"), loom::workgroup_size(1, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void entry(unsigned* output, const unsigned* input) {
  output[0u] = arithmetic::mix(input[0u]);
}
