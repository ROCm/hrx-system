// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_HIP_RUNTIME_H_
#define LOOMCXX_HIP_RUNTIME_H_

#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

#define __global__ LOOM_KERNEL
#define __device__ LOOM_DEVICE
#define __host__
#define __shared__ LOOM_WORKGROUP
#define __forceinline__ LOOM_FORCE_INLINE
#define threadIdx ::loom::kernel::workitem::id
#define blockIdx ::loom::kernel::workgroup::id
#define blockDim ::loom::kernel::workgroup::size
#define gridDim ::loom::kernel::workgroup::count
#define warpSize (::loom::kernel::subgroup::size())
#define __syncthreads() ::loom::kernel::workgroup::barrier()
#define __builtin_assume(condition) ::loom::assume(condition)
#define __builtin_inff() ::loom::infinity()
#define __builtin_amdgcn_rcpf(value) ::loom::reciprocal(value)

using uint3 = loom::kernel::uint3;

[[loom::device, loom::force_inline]] static inline float __shfl_xor(
    float value, int mask, int width = warpSize) {
  return loom::kernel::subgroup::shuffle_xor(value, mask, width).value;
}

[[loom::device, loom::force_inline]] static inline float __expf(float value) {
  return loom::scalar::approximate::expf(value);
}

// HIP spellings are ordinary source wrappers over Loom's typed vocabulary.
#define LOOMCXX_HIP_UNARY(name, operation)                                     \
  [[loom::device, loom::force_inline]] static inline float name(float value) { \
    return loom::scalar::operation(value);                                     \
  }
LOOMCXX_HIP_UNARY(expf, expf)
LOOMCXX_HIP_UNARY(exp2f, exp2f)
LOOMCXX_HIP_UNARY(expm1f, expm1f)
LOOMCXX_HIP_UNARY(__ocml_exp_f32, expf)
LOOMCXX_HIP_UNARY(logf, logf)
LOOMCXX_HIP_UNARY(log2f, log2f)
LOOMCXX_HIP_UNARY(log10f, log10f)
LOOMCXX_HIP_UNARY(log1pf, log1pf)
LOOMCXX_HIP_UNARY(tanhf, tanhf)
LOOMCXX_HIP_UNARY(sqrtf, sqrtf)
LOOMCXX_HIP_UNARY(rsqrtf, rsqrtf)
LOOMCXX_HIP_UNARY(sinf, sinf)
LOOMCXX_HIP_UNARY(cosf, cosf)
LOOMCXX_HIP_UNARY(fabsf, absf)
LOOMCXX_HIP_UNARY(erff, erff)
#undef LOOMCXX_HIP_UNARY

[[loom::device, loom::force_inline]] static inline float fmaxf(float lhs,
                                                               float rhs) {
  return loom::scalar::maxnumf(lhs, rhs);
}

[[loom::device, loom::force_inline]] static inline float fminf(float lhs,
                                                               float rhs) {
  return loom::scalar::minnumf(lhs, rhs);
}

[[loom::device, loom::force_inline]] static inline float powf(float lhs,
                                                              float rhs) {
  return loom::scalar::powf(lhs, rhs);
}

[[loom::device, loom::force_inline]] static inline float fmaf(float a, float b,
                                                              float c) {
  return loom::scalar::fmaf(a, b, c);
}

#endif  // LOOMCXX_HIP_RUNTIME_H_
