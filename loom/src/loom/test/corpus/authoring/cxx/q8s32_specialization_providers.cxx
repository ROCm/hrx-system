// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/target.h>
#include <loomcxx/vector.h>

#include "q8s32_specialization.h"

using Q8S32Float16 = float __attribute__((ext_vector_type(16)));
using Q8S32Partial8 = float __attribute__((ext_vector_type(8)));

LOOM_INLINE Q8S32BFloat16 q8s32_scaled_weights(Q8S32Codes16 weights,
                                               std::bfloat16_t scale) {
  auto wide = __builtin_convertvector(weights, Q8S32Float16);
  return __builtin_convertvector(wide * (float)scale, Q8S32BFloat16);
}

// Wave32 accumulates both halves into one vector bank before the horizontal
// reduction. This minimizes independent accumulator state.
LOOM_INLINE float q8s32_lane_dot_combined(Q8S32Codes16 weights_low,
                                          Q8S32Codes16 weights_high,
                                          Q8S32BFloat16 activations_low,
                                          Q8S32BFloat16 activations_high,
                                          std::bfloat16_t scale) {
  Q8S32Partial8 partial = {};
  partial = loom::vector::dot2f(q8s32_scaled_weights(weights_low, scale),
                                activations_low, partial);
  partial = loom::vector::dot2f(q8s32_scaled_weights(weights_high, scale),
                                activations_high, partial);
  return loom::vector::reduce::addf(partial, 0.0f);
}

// Wave64 keeps the halves independent through their vector reductions. The
// source expresses a different pressure/latency tradeoff with the same result.
LOOM_INLINE float q8s32_lane_dot_split(Q8S32Codes16 weights_low,
                                       Q8S32Codes16 weights_high,
                                       Q8S32BFloat16 activations_low,
                                       Q8S32BFloat16 activations_high,
                                       std::bfloat16_t scale) {
  Q8S32Partial8 low = {};
  Q8S32Partial8 high = {};
  low = loom::vector::dot2f(q8s32_scaled_weights(weights_low, scale),
                            activations_low, low);
  high = loom::vector::dot2f(q8s32_scaled_weights(weights_high, scale),
                             activations_high, high);
  return loom::vector::reduce::addf(low, 0.0f) +
         loom::vector::reduce::addf(high, 0.0f);
}

LOOM_TEMPLATE_DEF(q8s32_project)
[[loom::priority(40)]] float q8s32_wave32_values32(
    unsigned input_capacity, Q8S32Codes16 weights_low,
    Q8S32Codes16 weights_high, Q8S32BFloat16 activations_low,
    Q8S32BFloat16 activations_high, std::bfloat16_t scale)
    [[loom::where(loom::target::subgroup::size() == 32u &&
                  loom::predicate::multiple_of(input_capacity, 512u))]] {
  return loom::kernel::subgroup::reduce::addf<32>(q8s32_lane_dot_combined(
      weights_low, weights_high, activations_low, activations_high, scale));
}

LOOM_TEMPLATE_DEF(q8s32_project)
[[loom::priority(30)]] float q8s32_wave32_values16(
    unsigned input_capacity, Q8S32Codes16 weights_low,
    Q8S32Codes16 weights_high, Q8S32BFloat16 activations_low,
    Q8S32BFloat16 activations_high, std::bfloat16_t scale)
    [[loom::where(loom::target::subgroup::size() == 32u &&
                  loom::predicate::multiple_of(input_capacity, 768u))]] {
  return loom::kernel::subgroup::reduce::addf<16>(q8s32_lane_dot_combined(
      weights_low, weights_high, activations_low, activations_high, scale));
}

LOOM_TEMPLATE_DEF(q8s32_project)
[[loom::priority(40)]] float q8s32_wave64_values32(
    unsigned input_capacity, Q8S32Codes16 weights_low,
    Q8S32Codes16 weights_high, Q8S32BFloat16 activations_low,
    Q8S32BFloat16 activations_high, std::bfloat16_t scale)
    [[loom::where(loom::target::subgroup::size() == 64u &&
                  loom::predicate::multiple_of(input_capacity, 512u))]] {
  return loom::kernel::subgroup::reduce::addf<32>(q8s32_lane_dot_split(
      weights_low, weights_high, activations_low, activations_high, scale));
}

LOOM_TEMPLATE_DEF(q8s32_project)
[[loom::priority(30)]] float q8s32_wave64_values16(
    unsigned input_capacity, Q8S32Codes16 weights_low,
    Q8S32Codes16 weights_high, Q8S32BFloat16 activations_low,
    Q8S32BFloat16 activations_high, std::bfloat16_t scale)
    [[loom::where(loom::target::subgroup::size() == 64u &&
                  loom::predicate::multiple_of(input_capacity, 768u))]] {
  return loom::kernel::subgroup::reduce::addf<16>(q8s32_lane_dot_split(
      weights_low, weights_high, activations_low, activations_high, scale));
}

// The target-independent provider supplies an eight-lane implementation for
// modest capacities when no target-specific definition applies.
LOOM_TEMPLATE_DEF(q8s32_project)
[[loom::priority(1)]] float q8s32_generic_lanes8(unsigned input_capacity,
                                                 Q8S32Codes16 weights_low,
                                                 Q8S32Codes16 weights_high,
                                                 Q8S32BFloat16 activations_low,
                                                 Q8S32BFloat16 activations_high,
                                                 std::bfloat16_t scale)
    [[loom::where(loom::predicate::range(input_capacity, 256u, 2048u))]] {
  return loom::kernel::subgroup::reduce::addf<8>(q8s32_lane_dot_split(
      weights_low, weights_high, activations_low, activations_high, scale));
}
