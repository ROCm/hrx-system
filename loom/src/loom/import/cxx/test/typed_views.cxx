// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/encoding.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>
#include <loomcxx/view.h>

namespace loomt = loom::type;

using rows8 = loomt::view<loomt::shape<loomt::dynamic, 8>, const float>;
using Weight = loomt::float8_e4m3fn_t;

static rows8 preserve_view(rows8 source) { return source; }

template <class View>
static float read_view(View source) {
  return loom::view::load(source, 2, 3);
}

// Specializations retain different shape types through auto return deduction.
template <bool Strided>
static auto make_view(const float* input) {
  if constexpr (Strided) {
    auto layout = loom::encoding::layout::strided(11u, 1u);
    return loom::buffer::view<loomt::dynamic, 8>(input, {3}, layout);
  } else {
    auto layout = loom::encoding::layout::dense<2>();
    return loom::buffer::view<3, 8>(input, {}, layout);
  }
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void typed_view_static_layouts(const float* input, float* output,
                               unsigned input_origin) {
  loom::assume(input_origin < 16);
  auto dense_source = make_view<false>(input + input_origin);
  auto strided_source = preserve_view(make_view<true>(input + input_origin));
  output[0] = read_view(dense_source);
  output[1] = read_view(strided_source);
}

[[loom::kernel, loom::workgroup_size(8, 8, 1), loom::workgroup_count(1, 1, 1)]]
void typed_view_copy(const float* input, float* output, unsigned rows,
                     unsigned input_stride, unsigned input_origin,
                     unsigned output_origin) {
  loom::assume(rows < 9);
  loom::assume(input_stride < 32);
  loom::assume(input_origin < 16);
  loom::assume(output_origin < 16);

  auto input_layout = loom::encoding::layout::strided(input_stride, 1u);
  auto output_layout = loom::encoding::layout::dense<2>();
  auto source = preserve_view(loom::buffer::view<loomt::dynamic, 8>(
      input + input_origin, {rows}, input_layout));
  auto destination = loom::buffer::view<loomt::dynamic, 8>(
      output + output_origin, {rows}, output_layout);

  unsigned row = loom::workitem_id.y;
  unsigned column = loom::workitem_id.x;
  if (row < rows) {
    float value = loom::view::load(source, row, column);
    loom::view::store(value, destination, row, column);
  }
}

[[loom::kernel, loom::workgroup_size(16, 1, 1), loom::workgroup_count(1, 1, 1)]]
void typed_storage_rank3(
    [[loom::noalias, loom::assume_aligned(64)]] const Weight* input,
    [[loom::noalias, loom::assume_aligned(64)]] Weight* output, unsigned rows,
    unsigned tiles) {
  loom::assume(rows >= 1 && rows <= 65535);
  loom::assume(tiles >= 1 && tiles <= 65535);
  auto layout = loom::encoding::layout::dense<3>();
  auto schema = loom::encoding::define<loom::encoding::f8e4m3fn{
      .payload_elements = 16, .scale_group_elements = 16}>();
  auto storage = loom::encoding::define(layout, schema);
  auto source = loom::buffer::view<loomt::dynamic, loomt::dynamic, 16>(
      input, {rows, tiles}, storage);
  auto lane = loom::workitem_id.x;
  output[lane] = loom::view::load(source, 0, 0, lane);
}
