// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/fpu_state.h"
#include "iree/math/exponential.h"
#include "iree/math/roots.h"
#include "iree/math/trigonometry.h"
#include "iree/testing/benchmark.h"

namespace {

constexpr int64_t kBatchSize = 1024;

#define IREE_MATH_BENCHMARK(name, function, initial_value, scale, bias) \
  IREE_BENCHMARK_FN(name) {                                             \
    const iree_fpu_state_t fpu_state =                                  \
        iree_fpu_state_push(IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS |       \
                            IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST);      \
    float value = initial_value;                                        \
    while (iree_benchmark_keep_running(benchmark_state, kBatchSize)) {  \
      for (int64_t i = 0; i < kBatchSize; ++i) {                        \
        value = function(value) * scale + bias;                         \
      }                                                                 \
      iree_optimization_barrier(value);                                 \
    }                                                                   \
    iree_fpu_state_pop(fpu_state);                                      \
    return iree_ok_status();                                            \
  }                                                                     \
  IREE_BENCHMARK_REGISTER(name)

IREE_MATH_BENCHMARK(BM_Exp2F32Approx, iree_math_exp2_f32_approx, 0.125f, 0.125f,
                    0.25f);
IREE_MATH_BENCHMARK(BM_Log2F32Approx, iree_math_log2_f32_approx, 1.25f, 0.01f,
                    1.5f);
IREE_MATH_BENCHMARK(BM_ReciprocalF32Approx, iree_math_reciprocal_f32_approx,
                    1.25f, 0.01f, 1.5f);
IREE_MATH_BENCHMARK(BM_RsqrtF32Approx, iree_math_rsqrt_f32_approx, 1.25f, 0.01f,
                    1.5f);
IREE_MATH_BENCHMARK(BM_SqrtF32Approx, iree_math_sqrt_f32_approx, 1.25f, 0.01f,
                    1.5f);
IREE_MATH_BENCHMARK(BM_SinTurnsF32Approx, iree_math_sin_turns_f32_approx,
                    0.12345f, 0.001f, 0.12345f);
IREE_MATH_BENCHMARK(BM_CosTurnsF32Approx, iree_math_cos_turns_f32_approx,
                    0.12345f, 0.001f, 0.12345f);

#undef IREE_MATH_BENCHMARK

}  // namespace
