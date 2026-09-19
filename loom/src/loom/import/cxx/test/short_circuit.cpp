// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

__device__ __forceinline__ int record(int* output, unsigned index, int digit,
                                      int value) {
  output[index] = output[index] * 10 + digit;
  return value;
}

// Lanes beyond length must skip the input load and its effectful consumer.
__global__
    [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]] void
    short_circuit(const int* input, int* output, unsigned length) {
  unsigned lane = threadIdx.x;
  unsigned base = lane * 12u;
  output[base] = 0;
  output[base + 1u] = lane < length && record(output, base, 1, input[lane]);
  output[base + 2u] = 0;
  output[base + 3u] =
      lane >= length || record(output, base + 2u, 2, input[lane]);
  output[base + 4u] = 0;
  output[base + 5u] = record(output, base + 4u, 1, (int)(lane & 1u)) &&
                      record(output, base + 4u, 2, (int)(lane & 2u));
  output[base + 6u] = 0;
  output[base + 7u] = record(output, base + 6u, 1, (int)(lane & 1u)) ||
                      record(output, base + 6u, 2, (int)(lane & 2u));
  output[base + 8u] = 0;
  ((lane & 8u) && record(output, base + 8u, 4, 1));
  output[base + 9u] = (float(lane) - 31.0f) && (int(lane) - 32);
  // Each successful bound records its digit before evaluating the next bound.
  output[base + 10u] = 0;
  output[base + 11u] = lane < 64u && record(output, base + 10u, 1, 1) &&
                       lane < 32u && record(output, base + 10u, 2, 1) &&
                       lane < 16u && record(output, base + 10u, 3, 1) &&
                       lane < 8u && record(output, base + 10u, 4, 1) &&
                       lane < 4u && record(output, base + 10u, 5, 1) &&
                       lane < 2u && record(output, base + 10u, 6, 1) &&
                       lane < 1u;
}
