// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "q8s32_specialization.h"

#include <loomcxx/check.h>

[[loom::config("model.q8s32.input_capacity"),
  loom::where(loom::predicate::range(256u, 32768u) &&
              loom::predicate::multiple_of(256u))]]
extern const unsigned q8s32_input_capacity;

[[loom::kernel, loom::workgroup_size(32, 1, 1),
  loom::workgroup_count(1, 1, 1)]] void
q8s32_specialize(
    [[loom::noalias, loom::assume_aligned(64)]] const Q8S32Codes16* weights,
    [[loom::noalias,
      loom::assume_aligned(64)]] const Q8S32BFloat16* activations,
    [[loom::noalias, loom::assume_aligned(64)]] const std::bfloat16_t* scale,
    [[loom::noalias, loom::assume_aligned(64)]] float* output) {
  float result = loom::template_apply<q8s32_project>(
      q8s32_input_capacity, weights[0], weights[1], activations[0],
      activations[1], *scale);
  if (loom::kernel::subgroup::lane_id() == 0) {
    *output = result;
  }
}

LOOM_CHECK_CASE(q8s32_values32) {
  const auto weights = loom::check::fill<32, signed char>(1);
  const auto activations = loom::check::fill<32, unsigned short>(0x3F80);
  const auto scale = loom::check::fill<1, unsigned short>(0x3F80);
  const auto output = loom::check::fill<1, float>(0.0f);
  loom::check::launch<q8s32_specialize>(weights, activations, scale, output);
  // 32 exact products per lane reduced across a 32-lane cluster.
  loom::check::expect_bitwise(output, loom::check::fill<1, float>(1024.0f));
}

LOOM_CHECK_CASE(q8s32_values16) {
  const auto weights = loom::check::fill<32, signed char>(1);
  const auto activations = loom::check::fill<32, unsigned short>(0x3F80);
  const auto scale = loom::check::fill<1, unsigned short>(0x3F80);
  const auto output = loom::check::fill<1, float>(0.0f);
  loom::check::launch<q8s32_specialize>(weights, activations, scale, output);
  // 32 exact products per lane reduced across a 16-lane cluster.
  loom::check::expect_bitwise(output, loom::check::fill<1, float>(512.0f));
}

LOOM_CHECK_CASE(q8s32_values8) {
  const auto weights = loom::check::fill<32, signed char>(1);
  const auto activations = loom::check::fill<32, unsigned short>(0x3F80);
  const auto scale = loom::check::fill<1, unsigned short>(0x3F80);
  const auto output = loom::check::fill<1, float>(0.0f);
  loom::check::launch<q8s32_specialize>(weights, activations, scale, output);
  // 32 exact products per lane reduced across an 8-lane cluster.
  loom::check::expect_bitwise(output, loom::check::fill<1, float>(256.0f));
}

LOOM_CHECK_BENCHMARK(q8s32_values32_benchmark, q8s32_values32);
LOOM_CHECK_BENCHMARK(q8s32_values16_benchmark, q8s32_values16);
LOOM_CHECK_BENCHMARK(q8s32_values8_benchmark, q8s32_values8);
