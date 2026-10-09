// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "loom/codegen/low/immediate_fields.h"
#include "loom/ir/attribute.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"

namespace loom {
namespace {

TEST(WasmDescriptorFieldsTest, OptionalOffsetDistinguishesZeroFromAbsence) {
  const loom_named_attr_t entries[] = {{
      .name_id = 0,
      .reserved = 0,
      .value = loom_attr_i64(0),
  }};
  const auto attributes = loom_make_named_attr_slice(entries, 1);
  int evaluations = 0;
  const auto zero =
      loom_wasm_core_simd128_i32_load_offset((++evaluations, attributes));
  EXPECT_EQ(evaluations, 1);
  EXPECT_EQ(zero.kind, LOOM_ATTR_I64);
  EXPECT_EQ(zero.i64, 0);

  const auto absent = loom_wasm_core_simd128_i32_load_offset(
      (++evaluations, loom_named_attr_slice_empty()));
  EXPECT_EQ(evaluations, 2);
  EXPECT_TRUE(loom_attr_is_absent(absent));
}

TEST(WasmDescriptorFieldsTest, NamedPayloadsFollowDictionaryOrder) {
  // Canonical spelling order is hi64, lo64; wire order is lo64, hi64.
  const loom_named_attr_t entries[] = {
      {.name_id = 0, .reserved = 0, .value = loom_attr_i64(0x1234)},
      {.name_id = 1, .reserved = 0, .value = loom_attr_i64(0x5678)},
  };
  const auto attributes = loom_make_named_attr_slice(entries, 2);
  int evaluations = 0;
  EXPECT_EQ(
      loom_wasm_core_simd128_v128_const_lo64((++evaluations, attributes)).i64,
      0x5678);
  EXPECT_EQ(evaluations, 1);
  EXPECT_EQ(loom_wasm_core_simd128_v128_const_hi64(attributes).i64, 0x1234);
}

}  // namespace
}  // namespace loom
