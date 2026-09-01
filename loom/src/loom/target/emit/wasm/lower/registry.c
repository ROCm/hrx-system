// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/emit/wasm/contracts/sets.h"
#include "loom/target/emit/wasm/error_catalog.h"
#include "loom/target/emit/wasm/lower/lower.h"

static bool loom_wasm_type_is_i32_register(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      return true;
    default:
      return false;
  }
}

static bool loom_wasm_type_is_scalar_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_wasm_type_is_scalar_i64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

static bool loom_wasm_type_is_scalar_f64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F64;
}

static bool loom_wasm_source_type_supported(void* user_data,
                                            const loom_module_t* module,
                                            loom_type_t source_type) {
  (void)user_data;
  (void)module;
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(source_type);
  return scalar_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         scalar_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static bool loom_wasm_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static iree_status_t loom_wasm_make_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_I32, 1, out_type);
}

static iree_status_t loom_wasm_make_f32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_F32, 1, out_type);
}

static iree_status_t loom_wasm_make_i64_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_I64, 1, out_type);
}

static iree_status_t loom_wasm_make_f64_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_F64, 1, out_type);
}

static iree_status_t loom_wasm_make_v128_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_V128, 1, out_type);
}

static iree_status_t loom_wasm_map_type(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_type_t source_type,
                                        loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_wasm_type_is_i32_register(source_type)) {
    return loom_wasm_make_i32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_f32(source_type)) {
    return loom_wasm_make_f32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_i64(source_type)) {
    return loom_wasm_make_i64_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_f64(source_type)) {
    return loom_wasm_make_f64_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi1(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xf32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static iree_status_t loom_wasm_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_wasm_make_i32_register_type(context, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_source_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_wasm_map_type(user_data, context, source_function_op, source_type,
                            &out_argument->abi_type);
}

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .error_catalog = &loom_wasm_error_catalog,
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_wasm_map_argument, .user_data = NULL},
    .source_type_supported = {.fn = loom_wasm_source_type_supported,
                              .user_data = NULL},
    .contract_set = &loom_wasm_contract_set,
};

const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void) {
  return &kWasmLowLowerPolicy;
}

void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("wasm.core.simd128"),
          .policy = &kWasmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
