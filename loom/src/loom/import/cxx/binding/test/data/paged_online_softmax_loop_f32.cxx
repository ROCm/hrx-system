// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This body is the exact readable exporter output used by roundtrip coverage.
// clang-format off
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/scalar.h>
#include <loomcxx/vector.h>

using Float32x4 = float __attribute__((ext_vector_type(4)));

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(1, 1, 1)]]
void paged_online_softmax_loop_f32(loom::type::index token_count, [[loom::noalias, loom::assume_aligned(4)]] const float* query_storage, [[loom::noalias, loom::assume_aligned(4)]] const int* table_storage, [[loom::noalias, loom::assume_aligned(4)]] const float* key_storage, [[loom::noalias, loom::assume_aligned(4)]] const float* value_storage, [[loom::noalias, loom::assume_aligned(4)]] float* state_storage, [[loom::noalias, loom::assume_aligned(4)]] float* output_storage) {
  loom::assume(loom::predicate::range(token_count, 0, 32));
  Float32x4 initial_accumulator = 0.0f;
  auto query_view = loom::buffer::view<4>(query_storage, {}, loom::encoding::layout::dense<1>());
  auto table = loom::buffer::view<8>(table_storage, {}, loom::encoding::layout::dense<1>());
  auto state = loom::buffer::view<2>(state_storage, {}, loom::encoding::layout::dense<1>());
  auto output_view = loom::buffer::view<4>(output_storage, {}, loom::encoding::layout::dense<1>());
  Float32x4 query_vector = loom::vector::load<Float32x4>(query_view, loom::type::indices{loom::type::index{0}});
  float maximum = -1000000000.0f;
  float denominator = 0.0f;
  Float32x4 numerator = initial_accumulator;
  for (loom::type::index page_begin = loom::type::index{0}; page_begin < token_count; page_begin = page_begin + loom::type::index{4}) {
    loom::type::index logical_page = page_begin / loom::type::index{4};
    int page_id = loom::view::load(table, logical_page);
    bool present = page_id >= 0;
    if (present) {
      loom::type::index page_index = loom::index::cast(page_id);
      loom::assume(loom::predicate::range(page_index, 0, 2));
      loom::type::offset page_offset = loom::index::scale(page_index, loom::type::offset{64});
      maximum = maximum;
      denominator = denominator;
      numerator = numerator;
      for (int row = 0; row < 4; ++row) {
        loom::type::index token = page_begin + loom::index::cast(row);
        bool valid = token < token_count;
        if (valid) {
          loom::type::offset row_offset = loom::index::scale(loom::index::cast(row), loom::type::offset{16});
          loom::type::offset physical_offset = page_offset + row_offset;
          auto key_row = loom::buffer::view<4>(reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(key_storage) + static_cast<unsigned long long>(physical_offset)), {}, loom::encoding::layout::dense<1>());
          auto value_row = loom::buffer::view<4>(reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(value_storage) + static_cast<unsigned long long>(physical_offset)), {}, loom::encoding::layout::dense<1>());
          Float32x4 key = loom::vector::load<Float32x4>(key_row, loom::type::indices{loom::type::index{0}});
          Float32x4 value = loom::vector::load<Float32x4>(value_row, loom::type::indices{loom::type::index{0}});
          float score = loom::vector::dotf(query_vector, key, 0.0f);
          float updated_max = loom::scalar::maxnumf(maximum, score);
          float old_delta = maximum - updated_max;
          float score_delta = score - updated_max;
          float old_scale = loom::scalar::approximate::expf(old_delta);
          float weight = loom::scalar::approximate::expf(score_delta);
          [[loom::fastmath("contract")]] float rescaled_sum = denominator * old_scale;
          [[loom::fastmath("contract")]] denominator = rescaled_sum + weight;
          Float32x4 old_scale_vector = old_scale;
          Float32x4 weight_vector = weight;
          [[loom::fastmath("contract")]] Float32x4 rescaled_accumulator = numerator * old_scale_vector;
          [[loom::fastmath("contract")]] Float32x4 weighted_value = value * weight_vector;
          [[loom::fastmath("contract")]] numerator = rescaled_accumulator + weighted_value;
          maximum = updated_max;
        }
      }
    }
  }
  bool nonempty = denominator > 0.0f;
  Float32x4 normalized = Float32x4{};
  if (nonempty) {
    Float32x4 denominators = denominator;
    normalized = numerator / denominators;
  } else {
    normalized = initial_accumulator;
  }
  loom::view::store(maximum, state, loom::type::index{0});
  loom::view::store(denominator, state, loom::type::index{1});
  loom::vector::store(normalized, output_view, loom::type::indices{loom::type::index{0}});
}

// clang-format on
