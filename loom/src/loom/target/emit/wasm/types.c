// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/types.h"

#include "loom/target/arch/wasm/descriptors/descriptors.h"

loom_wasm_value_type_t loom_wasm_value_type_from_descriptor_register_class(
    uint16_t descriptor_reg_class_id) {
  switch (descriptor_reg_class_id) {
    case WASM_CORE_SIMD128_REG_CLASS_ID_I32:
      return LOOM_WASM_VALUE_TYPE_I32;
    case WASM_CORE_SIMD128_REG_CLASS_ID_I64:
      return LOOM_WASM_VALUE_TYPE_I64;
    case WASM_CORE_SIMD128_REG_CLASS_ID_F32:
      return LOOM_WASM_VALUE_TYPE_F32;
    case WASM_CORE_SIMD128_REG_CLASS_ID_F64:
      return LOOM_WASM_VALUE_TYPE_F64;
    case WASM_CORE_SIMD128_REG_CLASS_ID_V128:
      return LOOM_WASM_VALUE_TYPE_V128;
    default:
      IREE_ASSERT_UNREACHABLE("verified Wasm register class");
      IREE_BUILTIN_UNREACHABLE();
  }
}
