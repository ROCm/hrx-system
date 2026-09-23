// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

#include <stdfloat>

using Half4 = std::float16_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

// Each workgroup publishes 1024 embedding channels for one requested row.
[[loom::kernel, loom::workgroup_size(256, 1, 1),
  loom::workgroup_count(3, 2, 1)]]
void gather_rows([[loom::noalias,
                   loom::assume_aligned(64)]] const Half4* weights,
                 [[loom::noalias]] const unsigned* row_ids,
                 [[loom::noalias, loom::assume_aligned(64)]] Float4* output) {
  unsigned row = loom::workgroup_id.y;
  unsigned column = loom::workgroup_id.x * 256u + loom::workitem_id.x;
  unsigned source_row = row_ids[row];
  loom::assume(source_row < 2u);
  output[row * 768u + column] =
      __builtin_convertvector(weights[source_row * 768u + column], Float4);
}

[[loom::kernel, loom::workgroup_size(256, 1, 1),
  loom::workgroup_count(6, 1, 1)]]
void initialize_weights(Half4* weights) {
  unsigned index = loom::workgroup_id.x * 256u + loom::workitem_id.x;
  // Distinct lanes and row signs expose both row addressing and lane order.
  float sign = index < 768u ? 1.0f : -1.0f;
  weights[index] =
      Half4{std::float16_t(sign * 1.0f), std::float16_t(sign * 2.0f),
            std::float16_t(sign * 3.0f), std::float16_t(sign * 4.0f)};
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void initialize_indices(unsigned* indices) {
  indices[0] = 1u;
  indices[1] = 0u;
}

// An independent scalar check reads every output channel.
[[loom::kernel, loom::workgroup_size(256, 1, 1),
  loom::workgroup_count(24, 1, 1)]]
void check_rows(const float* output, unsigned* matches) {
  unsigned index = loom::workgroup_id.x * 256u + loom::workitem_id.x;
  float expected = float(index % 4u + 1u);
  if (index < 3072u) {
    expected = -expected;
  }
  matches[index] = output[index + 16u] == expected ? 1u : 0u;
}

LOOM_CHECK_CASE(noalias_embedding_rows) {
  const auto weights = loom::check::fill<std::float16_t, 6144>(0.0f);
  const auto weights_expected = loom::check::fill<std::float16_t, 6144>(0.0f);
  const auto indices = loom::check::fill<unsigned, 2>(0u);
  const auto output = loom::check::fill<float, 6176>(91.0f);
  const auto matches = loom::check::fill<unsigned, 6144>(0u);
  loom::check::launch<initialize_weights>(weights);
  loom::check::launch<initialize_weights>(weights_expected);
  loom::check::launch<initialize_indices>(indices);
  loom::check::launch<gather_rows>(weights, indices,
                                   loom::check::slice<6144>(output, 16));
  loom::check::launch<check_rows>(output, matches);
  loom::check::expect_bitwise(matches, loom::check::fill<unsigned, 6144>(1u));
  loom::check::expect_bitwise(loom::check::slice<16>(output, 0),
                              loom::check::fill<float, 16>(91.0f));
  loom::check::expect_bitwise(loom::check::slice<16>(output, 6160),
                              loom::check::fill<float, 16>(91.0f));
  loom::check::expect_bitwise(weights, weights_expected);
  loom::check::expect_bitwise(loom::check::slice<1>(indices, 0),
                              loom::check::fill<unsigned, 1>(1u));
  loom::check::expect_bitwise(loom::check::slice<1>(indices, 1),
                              loom::check::fill<unsigned, 1>(0u));
  loom::check::expect_event("device", "type", "asan_report", "count", 0);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void unmarked_alias([[loom::noalias]] const unsigned* input, unsigned* alias,
                    [[loom::noalias]] unsigned* output) {
  unsigned before = input[0];
  alias[0] = before + 5u;
  output[0] = input[0];
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void reassigned([[loom::noalias]] const unsigned* input,
                [[loom::noalias]] unsigned* output,
                const unsigned* replacement) {
  input = replacement;
  unsigned before = input[0];
  output[0] = before + 7u;
  output[1] = input[0];
}

LOOM_CHECK_CASE(noalias_preserves_possible_aliases) {
  const auto input = loom::check::fill<unsigned, 1>(17u);
  const auto output = loom::check::fill<unsigned, 2>(0u);
  const auto replacement = loom::check::fill<unsigned, 2>(31u);
  loom::check::launch<unmarked_alias>(input, input, output);
  loom::check::launch<reassigned>(input, replacement, replacement);
  loom::check::expect_bitwise(input, loom::check::fill<unsigned, 1>(22u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 0),
                              loom::check::fill<unsigned, 1>(22u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 1),
                              loom::check::fill<unsigned, 1>(0u));
  loom::check::expect_bitwise(replacement, loom::check::fill<unsigned, 2>(38u));
  loom::check::expect_event("device", "type", "asan_report", "count", 0);
}
