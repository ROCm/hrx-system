// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/encoding.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>
#include <loomcxx/vector.h>

#include <stdfloat>

using Words4 = unsigned __attribute__((ext_vector_type(4)));
using Words8 = unsigned __attribute__((ext_vector_type(8)));
using ScaleWord = unsigned __attribute__((ext_vector_type(1)));
using Fp8x32 = loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(32)));
using Fp8x64 = loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(64)));
using BFloat32 = std::bfloat16_t __attribute__((ext_vector_type(32)));
using BFloat64 = std::bfloat16_t __attribute__((ext_vector_type(64)));
using Float16 = float __attribute__((ext_vector_type(16)));
using Bits32 = unsigned __attribute__((ext_vector_type(32)));

struct Scales {
  // Four packed E8M0 bytes; each group consumes its corresponding byte.
  ScaleWord scale;
};

float mxfp4_group_dot(Words4 payload, ScaleWord scale, BFloat32 activations) {
  auto schema = loom::encoding::define<loom::encoding::f4e2m1{}>();
  auto weights = loom::vector::decode<BFloat32>(payload, schema, Scales{scale});
  Float16 partial = {};
  partial = loom::vector::dot2f(weights, activations, partial);
  return loom::vector::reduce::addf(partial, 0.0f);
}

float mxfp8_group_dot(Fp8x32 payload, ScaleWord scale, BFloat32 activations) {
  auto schema = loom::encoding::define<loom::encoding::f8e4m3fn{}>();
  auto weights = loom::vector::decode<BFloat32>(payload, schema, Scales{scale});
  Float16 partial = {};
  partial = loom::vector::dot2f(weights, activations, partial);
  return loom::vector::reduce::addf(partial, 0.0f);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp4_decode_dot(
    [[loom::noalias, loom::assume_aligned(64)]] const Words4* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat32* activations,
    [[loom::noalias]] float* output) {
  *output = mxfp4_group_dot(*payload, *scale, *activations);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp8_decode_dot(
    [[loom::noalias, loom::assume_aligned(64)]] const Fp8x32* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat32* activations,
    [[loom::noalias]] float* output) {
  *output = mxfp8_group_dot(*payload, *scale, *activations);
}

// Two groups in one vector retain their separate scale bytes through decoding.
[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp4_group_bits(
    [[loom::noalias, loom::assume_aligned(64)]] const Words8* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] Bits32* output) {
  auto schema =
      loom::encoding::define<loom::encoding::f4e2m1{.payload_elements = 64}>();
  auto weights =
      loom::vector::decode<BFloat64>(*payload, schema, Scales{*scale});
  *output = __builtin_bit_cast(Bits32, weights);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp8_group_bits(
    [[loom::noalias, loom::assume_aligned(64)]] const Fp8x64* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] Bits32* output) {
  auto schema = loom::encoding::define<loom::encoding::f8e4m3fn{
      .payload_elements = 64,
  }>();
  auto weights =
      loom::vector::decode<BFloat64>(*payload, schema, Scales{*scale});
  *output = __builtin_bit_cast(Bits32, weights);
}

// Require FP32 accumulation accuracy against independently evaluated products.
// The gamma_64 bound permits native grouping without selecting an output bit
// pattern from any particular target's arithmetic sequence.
constexpr double kGroupErrorFactor = 64.0 * 0x1p-24 / (1.0 - 64.0 * 0x1p-24);

LOOM_CHECK_CASE(mxfp4_dynamic_scale) {
  // E2M1 lanes repeat [0.5, 6, -2, -0.5, 3, -1, 1.5, -6].
  const auto payload = loom::check::fill<unsigned, 4>(0xF3A59C71u);
  // BF16 activations repeat [1, -2, 0.5, 4].
  const auto activations =
      loom::check::fill<unsigned long long, 8>(0x40803F00C0003F80ull);
  const auto identity = loom::check::fill<unsigned, 1>(127);
  const auto half = loom::check::fill<unsigned, 1>(126);
  const auto storage = loom::check::fill<float, 4>(999.0f);
  const auto first = loom::check::slice<1>(storage, 1);
  const auto second = loom::check::slice<1>(storage, 2);
  loom::check::launch<mxfp4_decode_dot>(payload, identity, activations, first);
  loom::check::launch<mxfp4_decode_dot>(payload, half, activations, second);
  // Products sum to -32.75 per eight lanes, with absolute sum 45.25.
  loom::check::expect_close(first, loom::check::fill<float, 1>(-131.0f),
                            181.0 * kGroupErrorFactor, 0.0, "different");
  loom::check::expect_close(second, loom::check::fill<float, 1>(-65.5f),
                            90.5 * kGroupErrorFactor, 0.0, "different");
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<float, 1>(999.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 3),
                              loom::check::fill<float, 1>(999.0f));
  loom::check::expect_bitwise(payload,
                              loom::check::fill<unsigned, 4>(0xF3A59C71u));
  loom::check::expect_bitwise(
      activations,
      loom::check::fill<unsigned long long, 8>(0x40803F00C0003F80ull));
  loom::check::expect_bitwise(identity, loom::check::fill<unsigned, 1>(127));
  loom::check::expect_bitwise(half, loom::check::fill<unsigned, 1>(126));
}

LOOM_CHECK_CASE(mxfp8_dynamic_scale) {
  // E4M3 lanes repeat [1.125, -1.25, 2.75, -4.5, 0.25, -0.125, 8, -16].
  const auto payload =
      loom::check::fill<unsigned long long, 4>(0xD850A028C943BA39ull);
  const auto activations =
      loom::check::fill<unsigned long long, 8>(0x40803F00C0003F80ull);
  const auto identity = loom::check::fill<unsigned, 1>(127);
  const auto half = loom::check::fill<unsigned, 1>(126);
  const auto storage = loom::check::fill<float, 4>(999.0f);
  const auto first = loom::check::slice<1>(storage, 1);
  const auto second = loom::check::slice<1>(storage, 2);
  loom::check::launch<mxfp8_decode_dot>(payload, identity, activations, first);
  loom::check::launch<mxfp8_decode_dot>(payload, half, activations, second);
  // Products sum to -72.5 per eight lanes, with absolute sum 91.5.
  loom::check::expect_close(first, loom::check::fill<float, 1>(-290.0f),
                            366.0 * kGroupErrorFactor, 0.0, "different");
  loom::check::expect_close(second, loom::check::fill<float, 1>(-145.0f),
                            183.0 * kGroupErrorFactor, 0.0, "different");
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<float, 1>(999.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 3),
                              loom::check::fill<float, 1>(999.0f));
  loom::check::expect_bitwise(
      payload, loom::check::fill<unsigned long long, 4>(0xD850A028C943BA39ull));
  loom::check::expect_bitwise(
      activations,
      loom::check::fill<unsigned long long, 8>(0x40803F00C0003F80ull));
  loom::check::expect_bitwise(identity, loom::check::fill<unsigned, 1>(127));
  loom::check::expect_bitwise(half, loom::check::fill<unsigned, 1>(126));
}

LOOM_CHECK_CASE(mxfp4_scale_bytes) {
  const auto payload = loom::check::fill<unsigned, 8>(0x11111111u);
  // 0.5 at E8M0 scales 0 and 128 gives exact BF16 subnormal 0x0020 and 1.0.
  const auto scales = loom::check::fill<unsigned, 1>(0x00008000u);
  const auto output = loom::check::fill<unsigned, 32>(0xFFFFFFFFu);
  loom::check::launch<mxfp4_group_bits>(payload, scales, output);
  loom::check::expect_bitwise(loom::check::slice<16>(output, 0),
                              loom::check::fill<unsigned, 16>(0x00200020u));
  loom::check::expect_bitwise(loom::check::slice<16>(output, 16),
                              loom::check::fill<unsigned, 16>(0x3F803F80u));
}

LOOM_CHECK_CASE(mxfp8_scale_bytes) {
  const auto payload = loom::check::fill<unsigned char, 64>(0x38);
  // 1.0 at E8M0 scales 0 and 255 gives exact BF16 subnormal 0x0040 and NaN.
  const auto scales = loom::check::fill<unsigned, 1>(0x0000FF00u);
  const auto output = loom::check::fill<unsigned, 32>(0xFFFFFFFFu);
  loom::check::launch<mxfp8_group_bits>(payload, scales, output);
  loom::check::expect_bitwise(loom::check::slice<16>(output, 0),
                              loom::check::fill<unsigned, 16>(0x00400040u));
  loom::check::expect_bitwise(loom::check::slice<16>(output, 16),
                              loom::check::fill<unsigned, 16>(0x7FC07FC0u));
}
