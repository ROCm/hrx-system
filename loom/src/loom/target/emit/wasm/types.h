// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly binary value-type helpers for Loom target-low emission.

#ifndef LOOM_TARGET_EMIT_WASM_TYPES_H_
#define LOOM_TARGET_EMIT_WASM_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_wasm_value_type_e {
  LOOM_WASM_VALUE_TYPE_I32 = 0x7F,
  LOOM_WASM_VALUE_TYPE_I64 = 0x7E,
  LOOM_WASM_VALUE_TYPE_F32 = 0x7D,
  LOOM_WASM_VALUE_TYPE_F64 = 0x7C,
  LOOM_WASM_VALUE_TYPE_V128 = 0x7B,
} loom_wasm_value_type_t;

// Resolves a descriptor-set-local Wasm register class ID to the Wasm binary
// value-type byte used in local declarations and allocated value emission.
iree_status_t loom_wasm_value_type_from_descriptor_register_class(
    uint16_t descriptor_reg_class_id, loom_wasm_value_type_t* out_value_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_TYPES_H_
