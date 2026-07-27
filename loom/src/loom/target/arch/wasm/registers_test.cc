// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/registers.h"

#include <string>

#include "iree/testing/gtest.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"

namespace loom {
namespace {

std::string DescriptorString(const loom_low_descriptor_set_t* descriptor_set,
                             loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
  return std::string(value.data, value.size);
}

TEST(WasmRegisterTypeResolverTest, ResolvesRealDescriptorIds) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();
  loom_low_register_type_resolver_t resolver =
      loom_low_register_type_resolver_for_descriptor_set(descriptor_set);

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = loom_low_register_type_resolver_try_resolve(
      &resolver,
      loom_low_register_type(descriptor_set->stable_id,
                             WASM_CORE_SIMD128_REG_CLASS_ID_V128, 1),
      &descriptor_register_class_id, &descriptor_register_class);
  ASSERT_TRUE(found);
  EXPECT_EQ(descriptor_register_class_id, WASM_CORE_SIMD128_REG_CLASS_ID_V128);
  ASSERT_NE(descriptor_register_class, nullptr);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             descriptor_register_class->name_string_offset),
            "wasm.v128");
}

}  // namespace
}  // namespace loom
