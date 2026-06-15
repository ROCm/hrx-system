// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/target/arch/spirv/abi.h"
#include "loom/target/arch/spirv/contracts/logical_core.h"
#include "loom/target/arch/spirv/contracts/logical_core_lower_rules.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/lower/matrix.h"
#include "loom/target/arch/spirv/lower/workgroup.h"
#include "loom/target/registers.h"

static iree_status_t loom_spirv_make_hal_buffer_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static iree_status_t loom_spirv_make_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, register_class_id, 1,
                                           out_type);
}

static bool loom_spirv_source_type_is_id(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_I1 ||
         scalar_type == LOOM_SCALAR_TYPE_I32 ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX;
}

static bool loom_spirv_source_type_is_offset64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_spirv_source_type_is_storage_scalar(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return true;
    default:
      return false;
  }
}

static bool loom_spirv_low_type_is_id_register(loom_type_t type) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID &&
         loom_low_register_type_class_id(type) ==
             SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID;
}

static iree_status_t loom_spirv_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_spirv_source_type_is_id(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, out_low_type);
  }
  if (loom_spirv_source_type_is_offset64(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, out_low_type);
  }
  if (loom_spirv_source_type_is_storage_scalar(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, out_low_type);
  }
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER,
        out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static iree_status_t loom_spirv_map_value(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_value_id,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  if (loom_type_is_vector(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, out_low_type);
  }
  return loom_spirv_map_type(user_data, context, source_op, source_type,
                             out_low_type);
}

static uint32_t loom_spirv_hal_binding_index(loom_low_lower_context_t* context,
                                             uint16_t source_argument_index) {
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

static iree_status_t loom_spirv_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)user_data;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t binding_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER,
        &binding_type));
    loom_type_t resource_source_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_spirv_make_hal_buffer_type(context, &resource_source_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = binding_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING,
        .resource_index =
            loom_spirv_hal_binding_index(context, source_argument_index),
        .resource_source_type = resource_source_type,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_spirv_map_type(user_data, context, source_function_op,
                             source_type, &out_argument->abi_type);
}

static iree_status_t loom_spirv_intern_abi_attr_keys(
    loom_module_t* module, loom_string_id_t* out_arg_key_id,
    loom_string_id_t* out_result_key_id) {
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV(LOOM_SPIRV_ABI_ARG_VALUE_TYPES_ATTR_NAME),
      out_arg_key_id));
  return loom_module_intern_string(
      module, IREE_SV(LOOM_SPIRV_ABI_RESULT_VALUE_TYPES_ATTR_NAME),
      out_result_key_id);
}

static bool loom_spirv_source_argument_lowers_to_direct_abi(
    const loom_target_bundle_t* bundle, loom_type_t source_type) {
  return bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL ||
         !loom_type_is_buffer(source_type);
}

static loom_attribute_t loom_spirv_make_abi_value_type_array_attr(
    int64_t* codes, iree_host_size_t count, bool has_payload) {
  if (!has_payload) {
    return loom_attr_absent();
  }
  IREE_ASSERT(count <= UINT16_MAX);
  return loom_attr_i64_array(codes, (uint16_t)count);
}

static iree_status_t loom_spirv_make_argument_value_type_codes(
    loom_low_lower_context_t* context, const loom_type_t* arg_types,
    iree_host_size_t arg_count, int64_t** out_codes, bool* out_has_payload) {
  *out_codes = NULL;
  *out_has_payload = false;
  if (arg_count == 0) {
    return iree_ok_status();
  }
  int64_t* codes = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, arg_count, sizeof(*codes), (void**)&codes));
  memset(codes, 0, arg_count * sizeof(*codes));

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &source_argument_count);
  iree_host_size_t direct_argument_index = 0;
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(module, source_arguments[i]);
    if (!loom_spirv_source_argument_lowers_to_direct_abi(bundle, source_type)) {
      continue;
    }
    if (direct_argument_index >= arg_count) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "SPIR-V ABI argument mapping exceeded the "
                              "mapped low signature");
    }
    if (loom_spirv_low_type_is_id_register(arg_types[direct_argument_index])) {
      loom_spirv_value_type_t value_type = {0};
      if (!loom_spirv_abi_value_type_from_source_type(source_type,
                                                      &value_type) ||
          !loom_spirv_abi_value_type_encode(value_type,
                                            &codes[direct_argument_index])) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "SPIR-V low ABI could not encode a mapped "
                                "direct argument payload");
      }
      *out_has_payload = true;
    }
    ++direct_argument_index;
  }
  if (direct_argument_index != arg_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "SPIR-V ABI argument mapping did not cover the "
                            "mapped low signature");
  }

  *out_codes = codes;
  return iree_ok_status();
}

static iree_status_t loom_spirv_make_result_value_type_codes(
    loom_low_lower_context_t* context, const loom_type_t* result_types,
    iree_host_size_t result_count, int64_t** out_codes, bool* out_has_payload) {
  *out_codes = NULL;
  *out_has_payload = false;
  if (result_count == 0) {
    return iree_ok_status();
  }
  int64_t* codes = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, result_count, sizeof(*codes), (void**)&codes));
  memset(codes, 0, result_count * sizeof(*codes));

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t* source_results =
      loom_op_const_results(loom_low_lower_context_source_function(context).op);
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    if (!loom_spirv_low_type_is_id_register(result_types[i])) {
      continue;
    }
    const loom_type_t source_type =
        loom_module_value_type(module, source_results[i]);
    loom_spirv_value_type_t value_type = {0};
    if (!loom_spirv_abi_value_type_from_source_type(source_type, &value_type) ||
        !loom_spirv_abi_value_type_encode(value_type, &codes[i])) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "SPIR-V low ABI could not encode a mapped "
                              "result payload");
    }
    *out_has_payload = true;
  }

  *out_codes = codes;
  return iree_ok_status();
}

static iree_status_t loom_spirv_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_named_attr_slice_t* out_abi_layout) {
  (void)user_data;
  *out_abi_layout = loom_named_attr_slice_empty();
  loom_string_id_t arg_key_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t result_key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_spirv_intern_abi_attr_keys(
      loom_low_lower_context_module(context), &arg_key_id, &result_key_id));

  int64_t* arg_codes = NULL;
  bool has_arg_payload = false;
  IREE_RETURN_IF_ERROR(loom_spirv_make_argument_value_type_codes(
      context, arg_types, arg_count, &arg_codes, &has_arg_payload));

  int64_t* result_codes = NULL;
  bool has_result_payload = false;
  if (layout_kind == LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC) {
    IREE_RETURN_IF_ERROR(loom_spirv_make_result_value_type_codes(
        context, result_types, result_count, &result_codes,
        &has_result_payload));
  }

  loom_attribute_t arg_attr = loom_spirv_make_abi_value_type_array_attr(
      arg_codes, arg_count, has_arg_payload);
  loom_attribute_t result_attr = loom_spirv_make_abi_value_type_array_attr(
      result_codes, result_count, has_result_payload);

  const iree_host_size_t added_count =
      (has_arg_payload ? 1 : 0) + (has_result_payload ? 1 : 0);
  if (added_count == 0) {
    return iree_ok_status();
  }

  loom_named_attr_t* entries = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, added_count, sizeof(*entries), (void**)&entries));
  iree_host_size_t entry_index = 0;
  if (has_arg_payload) {
    entries[entry_index++] = (loom_named_attr_t){
        .name_id = arg_key_id,
        .value = arg_attr,
    };
  }
  if (has_result_payload) {
    entries[entry_index++] = (loom_named_attr_t){
        .name_id = result_key_id,
        .value = result_attr,
    };
  }
  *out_abi_layout = loom_make_named_attr_slice(entries, added_count);
  return iree_ok_status();
}

static const loom_low_lower_rule_set_t* const kSpirvRuleSets[] = {
    &loom_spirv_logical_core_lower_rule_set,
};

static const loom_target_contract_binding_t kSpirvContractBindings[] = {
    {&loom_spirv_logical_core_contract_fragment, 0},
};

static iree_status_t loom_spirv_preselect_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  return loom_spirv_select_workgroup_plan(context, source_op, out_plan);
}

static iree_status_t loom_spirv_emit_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  return loom_spirv_lower_workgroup_op(context, source_op, plan);
}

static const loom_low_lower_policy_t kSpirvLowLowerPolicy = {
    .name = IREE_SVL("spirv-logical-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_spirv_map_type, .user_data = NULL},
    .map_value = {.fn = loom_spirv_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_spirv_map_argument, .user_data = NULL},
    .map_abi_layout = {.fn = loom_spirv_map_abi_layout, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kSpirvRuleSets),
            .values = kSpirvRuleSets,
        },
    .contract_bindings = kSpirvContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kSpirvContractBindings),
    .descriptor_matrix =
        {
            .options = loom_spirv_descriptor_matrix_options,
            .query = loom_spirv_descriptor_matrix_query,
            .attrs = NULL,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_spirv_preselect_op, .user_data = NULL},
    .emit_op = {.fn = loom_spirv_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_spirv_low_lower_policy(void) {
  return &kSpirvLowLowerPolicy;
}

void loom_spirv_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("spirv.logical.core"),
          .policy = &kSpirvLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
