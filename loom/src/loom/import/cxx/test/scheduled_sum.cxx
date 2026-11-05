// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <loomcxx/check.h>
#include <loomcxx/predicate.h>
#include <loomcxx/view.h>

struct ScheduledSumConfiguration {
  unsigned columns;
  loom::type::index input_count;
};

LOOM_INLINE static ScheduledSumConfiguration make_configuration(
    unsigned columns, unsigned input_count) {
  return {columns, loom::index::cast(input_count)};
}

static ScheduledSumConfiguration scheduled_sum_configuration(
    loom::check::ordinal trial) [[loom::where(loom::predicate::range(
    loom::predicate::result<ScheduledSumConfiguration>().columns, 0u, 33u))]] {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return make_configuration(0u, 7u);
  }
  if (trial_value == 1u) {
    return make_configuration(1u, 7u);
  }
  if (trial_value == 2u) {
    return make_configuration(2u, 14u);
  }
  if (trial_value == 3u) {
    return make_configuration(5u, 35u);
  }
  if (trial_value == 4u) {
    return make_configuration(17u, 119u);
  }
  return make_configuration(33u, 231u);
}

__device__ __forceinline__ int sum(const int* input, unsigned row,
                                   unsigned columns, unsigned factor,
                                   unsigned depth) {
  int total = 0;
  [[loom::unroll(factor), loom::pipeline(depth), loom::schedule("linear")]]
  for (unsigned column = 0; column < columns; ++column) {
    total += input[row * columns + column];
  }
  return total;
}

// Each entry exercises one schedule through the same source expressions.
#define SCHEDULED_SUM(unroll, pipeline_depth)                               \
  __global__                                                                \
      [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]    \
      void scheduled_sum_unroll_##unroll##_depth_##pipeline_depth(          \
          const int* input, int* output, unsigned rows, unsigned columns) { \
    unsigned row = threadIdx.x;                                             \
    if (row < rows) {                                                       \
      unsigned factor = unroll + 1u;                                        \
      --factor;                                                             \
      unsigned depth = blockDim.x / 64u * pipeline_depth;                   \
      output[row] = sum(input, row, columns, factor, depth);                \
    }                                                                       \
  }

SCHEDULED_SUM(1, 1)
SCHEDULED_SUM(1, 2)
SCHEDULED_SUM(3, 1)
SCHEDULED_SUM(3, 2)

static void scheduled_sum_oracle(loom::type::buffer<int> input,
                                 loom::type::buffer<int> output, unsigned rows,
                                 unsigned columns) {
  loom::assume(rows == 7u);
  loom::assume(columns <= 33u);
  unsigned input_count = rows * columns;
  auto input_extent = loom::index::cast(input_count);
  auto output_values =
      loom::buffer::view<7>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned row = 0; row < rows; ++row) {
    int total = 0;
    for (unsigned column = 0; column < columns; ++column) {
      unsigned position = row * columns + column;
      auto input_position = loom::index::cast(position);
      loom::assume(loom::predicate::lt(input_position, input_extent));
      auto input_values = loom::buffer::view<loom::type::dynamic>(
          input, loom::type::indices{input_extent},
          loom::encoding::layout::dense<1>());
      total += loom::view::load(input_values, input_position);
    }
    loom::view::store(total, output_values, row);
  }
}

#define SCHEDULED_SUM_TRIAL(kernel)                                            \
  loom::check::trial<6>([](loom::check::ordinal trial,                         \
                           loom::check::entropy entropy) {                     \
    const auto configuration =                                                 \
        loom::check::generate<scheduled_sum_configuration>(trial);             \
    const auto input = loom::check::iota(configuration.input_count, -100, 37); \
    const auto storage = loom::check::fill<39, int>(-123);                     \
    const auto output = loom::check::slice<7>(storage, 16);                    \
    loom::check::compare<kernel, scheduled_sum_oracle>(                        \
        input, output, 7u, configuration.columns, [&] {                        \
          loom::check::expect_bitwise(input, input);                           \
          loom::check::expect_bitwise(storage, storage);                       \
        });                                                                    \
  })

LOOM_CHECK_SCENARIO(scheduled_sum_values) {
  SCHEDULED_SUM_TRIAL(scheduled_sum_unroll_1_depth_1);
  SCHEDULED_SUM_TRIAL(scheduled_sum_unroll_1_depth_2);
  SCHEDULED_SUM_TRIAL(scheduled_sum_unroll_3_depth_1);
  SCHEDULED_SUM_TRIAL(scheduled_sum_unroll_3_depth_2);
}

#undef SCHEDULED_SUM_TRIAL
#undef SCHEDULED_SUM
