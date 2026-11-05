// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_HIP_FP16_H_
#define LOOMCXX_HIP_FP16_H_

using half = _Float16;
using __half = _Float16;

[[loom::device, loom::inline]] static inline half __float2half_rn(float value) {
  return (half)value;
}

[[loom::device, loom::inline]] static inline float __half2float(half value) {
  return (float)value;
}

#endif  // LOOMCXX_HIP_FP16_H_
