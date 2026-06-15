// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/types.h"

#include <inttypes.h>

#include "loom/target/arch/wasm/descriptors/descriptors.h"

iree_status_t loom_wasm_value_type_from_descriptor_register_class(
    uint16_t descriptor_reg_class_id, loom_wasm_value_type_t* out_value_type) {
  switch (descriptor_reg_class_id) {
    case WASM_CORE_SIMD128_REG_CLASS_ID_I32:
      *out_value_type = LOOM_WASM_VALUE_TYPE_I32;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_I64:
      *out_value_type = LOOM_WASM_VALUE_TYPE_I64;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_F32:
      *out_value_type = LOOM_WASM_VALUE_TYPE_F32;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_F64:
      *out_value_type = LOOM_WASM_VALUE_TYPE_F64;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_V128:
      *out_value_type = LOOM_WASM_VALUE_TYPE_V128;
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "unsupported Wasm descriptor register class ID %" PRIu16,
          descriptor_reg_class_id);
  }
}
