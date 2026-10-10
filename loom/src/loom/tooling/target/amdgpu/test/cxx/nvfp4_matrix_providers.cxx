// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/encoding.h>
#include <loomcxx/target/amdgpu.h>
#include <loomcxx/vector.h>

#include <stdfloat>

#include "nvfp4_matrix.h"

namespace fragment = loom::vector::fragment;

using NvFp4Half4 = std::float16_t __attribute__((ext_vector_type(4)));
using NvFp4Half8 = std::float16_t __attribute__((ext_vector_type(8)));
using NvFp4Half16 = std::float16_t __attribute__((ext_vector_type(16)));
using NvFp4Payload4 = unsigned __attribute__((ext_vector_type(1)));
using NvFp4Float4 = float __attribute__((ext_vector_type(4)));
using NvFp4Float8 = float __attribute__((ext_vector_type(8)));
using NvFp4Schema = loom::type::encoding<loom::encoding::role::schema>;

struct NvFp4Parameters {
  NvFp4Scale scale;
};

constexpr loom::amdgpu::target nvfp4_wave32_target{
    .kind = "gfx11-generic",
};

constexpr loom::amdgpu::target nvfp4_wave64_target{
    .kind = "gfx9-4-generic",
};

LOOM_FORCE_INLINE NvFp4Schema nvfp4_schema() {
  return loom::encoding::define<loom::encoding::f4e2m1{
      .payload_elements = 16,
      .payload_registers = 2,
      .scale_format = loom::encoding::numeric_format::f8e4m3fn,
      .scale_group_elements = 16,
  }>();
}

LOOM_TEMPLATE_DEF(nvfp4_matrix_tile)
[[loom::target(nvfp4_wave32_target), loom::priority(20)]]
void nvfp4_matrix_tile_wave32(NvFp4Payload lhs_payload, NvFp4Scale lhs_scale,
                              NvFp4Payload rhs_payload, NvFp4Scale rhs_scale,
                              float* output)
    [[loom::where(loom::target::subgroup::size() == 32u)]] {
  auto schema = nvfp4_schema();
  auto lhs_data = loom::vector::decode<NvFp4Half16>(lhs_payload, schema,
                                                    NvFp4Parameters{lhs_scale});
  auto rhs_data = loom::vector::decode<NvFp4Half16>(rhs_payload, schema,
                                                    NvFp4Parameters{rhs_scale});
  auto lhs =
      fragment::attach<fragment::role::lhs>(lhs_data, fragment::shape{16, 16});
  auto rhs =
      fragment::attach<fragment::role::rhs>(rhs_data, fragment::shape{16, 16});
  NvFp4Float8 zero = {};
  auto init =
      fragment::attach<fragment::role::init>(zero, fragment::shape{16, 16});
  auto result = loom::vector::mma(lhs, rhs, init);
  auto output_view = loom::buffer::view<16, 16>(
      output, {}, loom::encoding::layout::dense<2>());
  fragment::store<fragment::role::result>(result, output_view, {0, 0},
                                          fragment::shape{16, 16});
}

LOOM_TEMPLATE_DEF(nvfp4_matrix_tile)
[[loom::target(nvfp4_wave64_target), loom::priority(20)]]
void nvfp4_matrix_tile_wave64(NvFp4Payload lhs_payload, NvFp4Scale lhs_scale,
                              NvFp4Payload rhs_payload, NvFp4Scale rhs_scale,
                              float* output)
    [[loom::where(loom::target::subgroup::size() == 64u)]] {
  auto schema = loom::encoding::define<loom::encoding::f4e2m1{
      .payload_elements = 8,
      .payload_registers = 1,
      .scale_format = loom::encoding::numeric_format::f8e4m3fn,
      .scale_group_elements = 8,
  }>();
  unsigned lane = loom::kernel::subgroup::lane_id();
  // Each 16-lane group supplies one packed 16-bit quarter of the logical
  // payload to the wave64 MFMA's four f16 values per lane.
  unsigned payload_word = lane >> 5u;
  unsigned payload_shift = lane & 16u;
  NvFp4Payload4 lhs_selected = {lhs_payload[payload_word] >> payload_shift};
  NvFp4Payload4 rhs_selected = {rhs_payload[payload_word] >> payload_shift};
  auto lhs_decoded = loom::vector::decode<NvFp4Half8>(
      lhs_selected, schema, NvFp4Parameters{lhs_scale});
  auto rhs_decoded = loom::vector::decode<NvFp4Half8>(
      rhs_selected, schema, NvFp4Parameters{rhs_scale});
  auto lhs_data = loom::vector::slice<NvFp4Half4>(lhs_decoded, 0);
  auto rhs_data = loom::vector::slice<NvFp4Half4>(rhs_decoded, 0);
  auto lhs =
      fragment::attach<fragment::role::lhs>(lhs_data, fragment::shape{16, 16});
  auto rhs =
      fragment::attach<fragment::role::rhs>(rhs_data, fragment::shape{16, 16});
  NvFp4Float4 zero = {};
  auto init =
      fragment::attach<fragment::role::init>(zero, fragment::shape{16, 16});
  auto result = loom::vector::mma(lhs, rhs, init);
  auto output_view = loom::buffer::view<16, 16>(
      output, {}, loom::encoding::layout::dense<2>());
  fragment::store<fragment::role::result>(result, output_view, {0, 0},
                                          fragment::shape{16, 16});
}
