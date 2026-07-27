// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/target/arch/llvmir/contracts/generic_core.h"
#include "loom/target/arch/llvmir/contracts/generic_core_lower_rules.h"
#include "loom/target/arch/llvmir/descriptors/descriptors.h"
#include "loom/target/arch/llvmir/lower/lower.h"

static iree_status_t loom_llvmir_make_hal_buffer_resource_source_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static bool loom_llvmir_scalar_register_class_for_type(
    loom_type_t type, uint16_t* out_register_class_id,
    uint32_t* out_unit_count) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  *out_unit_count = 1;
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_I1:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I1;
      return true;
    case LOOM_SCALAR_TYPE_I8:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I32;
      return true;
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I64;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_F16;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_BF16;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_F32;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_F64;
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_vector_register_class_for_type(
    loom_type_t type, uint16_t* out_register_class_id,
    uint32_t* out_unit_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count <= 0 || lane_count > UINT32_MAX) {
    return false;
  }
  *out_unit_count = (uint32_t)lane_count;
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_I1:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I1;
      return true;
    case LOOM_SCALAR_TYPE_I8:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I32;
      return true;
    case LOOM_SCALAR_TYPE_I64:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_I64;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_F16;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_BF16;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_F32;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_F64;
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_register_class_for_source_type(
    loom_type_t source_type, uint16_t* out_register_class_id,
    uint32_t* out_unit_count) {
  if (loom_llvmir_scalar_register_class_for_type(
          source_type, out_register_class_id, out_unit_count)) {
    return true;
  }
  if (loom_llvmir_vector_register_class_for_type(
          source_type, out_register_class_id, out_unit_count)) {
    return true;
  }
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type)) {
    *out_register_class_id = LLVMIR_GENERIC_CORE_REG_CLASS_ID_PTR;
    *out_unit_count = 1;
    return true;
  }
  return false;
}

static iree_status_t loom_llvmir_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  uint16_t register_class_id = LOOM_LOW_REG_CLASS_NONE;
  uint32_t unit_count = 0;
  if (loom_llvmir_register_class_for_source_type(
          source_type, &register_class_id, &unit_count)) {
    return loom_low_lower_make_register_type(context, register_class_id,
                                             unit_count, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static uint32_t loom_llvmir_kernel_binding_resource_index(
    loom_low_lower_context_t* context, uint16_t source_argument_index) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &argument_count);
  uint32_t resource_index = 0;
  for (uint16_t i = 0; i < source_argument_index && i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), argument_ids[i]);
    if (loom_type_is_buffer(type)) {
      ++resource_index;
    }
  }
  return resource_index;
}

static iree_status_t loom_llvmir_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t binding_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, LLVMIR_GENERIC_CORE_REG_CLASS_ID_PTR, 1, &binding_type));
    loom_type_t resource_source_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_llvmir_make_hal_buffer_resource_source_type(
        context, &resource_source_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = binding_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING,
        .resource_index = loom_llvmir_kernel_binding_resource_index(
            context, source_argument_index),
        .resource_source_type = resource_source_type,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_llvmir_map_type(user_data, context, source_function_op,
                              source_type, &out_argument->abi_type);
}

static const loom_low_lower_rule_set_t* const kLlvmirGenericRuleSets[] = {
    &loom_llvmir_generic_core_lower_rule_set,
};

static const loom_target_contract_binding_t kLlvmirGenericContractBindings[] = {
    {&loom_llvmir_generic_core_contract_fragment, 0},
};

static const loom_low_lower_policy_t kLlvmirGenericLowLowerPolicy = {
    .name = IREE_SVL("llvmir-generic-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_llvmir_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_llvmir_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kLlvmirGenericRuleSets),
            .values = kLlvmirGenericRuleSets,
        },
    .contract_bindings = kLlvmirGenericContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kLlvmirGenericContractBindings),
};

const loom_low_lower_policy_t* loom_llvmir_generic_low_lower_policy(void) {
  return &kLlvmirGenericLowLowerPolicy;
}

void loom_llvmir_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("llvmir.generic.core"),
          .policy = &kLlvmirGenericLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
