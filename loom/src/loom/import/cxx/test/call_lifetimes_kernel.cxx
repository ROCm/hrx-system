// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[[loom::noinline]] static unsigned increment(unsigned* value) {
  return ++*value;
}

[[loom::noinline]] static void publish(unsigned* output, unsigned value) {
  *output = value;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void call_lifetimes_kernel(const unsigned* parameters, unsigned* output) {
  for (unsigned index = 0; index < parameters[0]; ++index) {
    // Dynamic indexing keeps an unpassed array addressable across a call.
    unsigned values[2] = {index, index + 7};
    publish(output + index, values[parameters[1] & 1]);
    // A passed scalar is read and written by the callee before it returns.
    unsigned counter = index;
    unsigned updated = increment(&counter);
    output[parameters[0] + index] = updated + counter;
  }
}
