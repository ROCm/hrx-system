// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/spirv/contracts/logical_core.h"
#include "loom/target/arch/spirv/contracts/logical_core_lower_rules.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/lower/matrix.h"
#include "loom/target/arch/spirv/lower/workgroup.h"
#include "loom/target/arch/spirv/ops/types.h"
#include "loom/target/arch/spirv/value_types.h"
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

static iree_status_t loom_spirv_make_typed_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    loom_type_t value_type, loom_type_t* out_type) {
  return loom_low_lower_make_typed_register_type(context, register_class_id, 1,
                                                 value_type, out_type);
}

static bool loom_spirv_source_type_is_id(loom_type_t type) {
  loom_spirv_value_type_t value_type = {0};
  return loom_spirv_value_type_from_loom_type(type, &value_type);
}

static bool loom_spirv_source_type_is_offset64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_spirv_source_type_is_fp8(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         scalar_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static bool loom_spirv_source_type_supported(void* user_data,
                                             const loom_module_t* module,
                                             loom_type_t source_type) {
  (void)user_data;
  (void)module;
  return loom_spirv_source_type_is_fp8(source_type);
}

static bool loom_spirv_source_value_fragment(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value_id,
    loom_vector_fragment_fact_t* out_fragment) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  return loom_vector_fragment_fact_query_value_facts(&fact_table->context,
                                                     facts, out_fragment) &&
         loom_vector_fragment_fact_has_matrix_shape(*out_fragment);
}

static bool loom_spirv_fragment_dimension(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    uint16_t* out_dimension) {
  int64_t dimension = 0;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &dimension) ||
      dimension <= 0 || dimension > UINT16_MAX) {
    return false;
  }
  *out_dimension = (uint16_t)dimension;
  return true;
}

static bool loom_spirv_fragment_role(
    loom_vector_fragment_fact_t fragment,
    loom_contract_operand_role_t* out_contract_role,
    loom_spirv_cooperative_matrix_use_t* out_use) {
  if (fragment.role_flags == LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS) {
    *out_contract_role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
    *out_use = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR;
    return true;
  }
  if (fragment.role_flags == LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS) {
    *out_contract_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
    *out_use = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR;
    return true;
  }
  if (loom_vector_fragment_fact_is_accumulator_like(fragment)) {
    *out_contract_role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
    *out_use = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR;
    return true;
  }
  return false;
}

static bool loom_spirv_fragment_matrix_payload(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_vector_fragment_fact_t fragment,
    uint16_t* out_rows, uint16_t* out_columns,
    loom_spirv_scalar_type_t* out_component_type,
    loom_spirv_cooperative_matrix_use_t* out_use) {
  if (!loom_spirv_fragment_dimension(
          fact_table, loom_vector_fragment_fact_row_value(fragment),
          out_rows) ||
      !loom_spirv_fragment_dimension(
          fact_table, loom_vector_fragment_fact_column_value(fragment),
          out_columns)) {
    return false;
  }
  if (fragment.shape_rank == 3) {
    uint16_t block_count = 0;
    if (!loom_spirv_fragment_dimension(
            fact_table, loom_vector_fragment_fact_block_value(fragment),
            &block_count) ||
        block_count != 1) {
      return false;
    }
  }

  loom_contract_operand_role_t contract_role =
      LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  if (!loom_spirv_fragment_role(fragment, &contract_role, out_use)) {
    return false;
  }
  loom_contract_operand_t contract_operand = {0};
  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  if (!loom_contract_vector_operand_from_fragment(
          module, source_value_id, fragment, contract_role, &contract_operand,
          &rejection_bits) ||
      !loom_spirv_matrix_component_type(contract_operand.numeric_type,
                                        out_component_type)) {
    return false;
  }
  return true;
}

static iree_status_t loom_spirv_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_spirv_source_type_is_fp8(source_type)) {
    return loom_spirv_make_typed_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID,
        loom_type_scalar(LOOM_SCALAR_TYPE_I8), out_low_type);
  }
  if (loom_spirv_source_type_is_id(source_type)) {
    return loom_spirv_make_typed_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, source_type, out_low_type);
  }
  if (loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
      !loom_type_dim_is_dynamic_at(source_type, 0) &&
      loom_type_dim_static_size_at(source_type, 0) == 1) {
    return loom_spirv_map_type(
        user_data, context, source_op,
        loom_type_scalar(loom_type_element_type(source_type)), out_low_type);
  }
  if (loom_spirv_source_type_is_offset64(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, out_low_type);
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
    if (loom_func_return_isa(source_op)) {
      return loom_low_lower_emit_source_type_unsupported(
          context, source_op, IREE_SV("source"), source_type);
    }
    loom_vector_fragment_fact_t fragment = {0};
    const loom_value_fact_table_t* fact_table =
        loom_low_lower_context_fact_table(context);
    if (loom_spirv_source_value_fragment(fact_table, source_value_id,
                                         &fragment)) {
      uint16_t rows = 0;
      uint16_t columns = 0;
      loom_spirv_scalar_type_t component_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
      loom_spirv_cooperative_matrix_use_t use =
          LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MAX;
      if (loom_spirv_fragment_matrix_payload(
              loom_low_lower_context_module(context), fact_table,
              source_value_id, fragment, &rows, &columns, &component_type,
              &use)) {
        loom_type_t matrix_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_spirv_cooperative_matrix_type_make(
            loom_low_lower_context_module(context), rows, columns,
            component_type, loom_spirv_matrix_scope(), use, &matrix_type));
        return loom_spirv_make_typed_register_type(
            context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, matrix_type,
            out_low_type);
      }
      return loom_spirv_make_register_type(
          context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, out_low_type);
    }
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
  if (loom_type_is_vector(source_type)) {
    return loom_low_lower_emit_source_type_unsupported(
        context, source_function_op, IREE_SV("argument"), source_type);
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_spirv_map_type(user_data, context, source_function_op,
                             source_type, &out_argument->abi_type);
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

static void loom_spirv_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  loom_spirv_mark_workgroup_plan_storage_demands(context, source_op, plan);
}

static const loom_low_lower_policy_t kSpirvLowLowerPolicy = {
    .name = IREE_SVL("spirv-logical-lower"),
    .error_catalog = &loom_spirv_error_catalog,
    .map_type = {.fn = loom_spirv_map_type, .user_data = NULL},
    .map_value = {.fn = loom_spirv_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_spirv_map_argument, .user_data = NULL},
    .source_type_supported = {.fn = loom_spirv_source_type_supported,
                              .user_data = NULL},
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
    .mark_plan_storage_demands =
        {
            .fn = loom_spirv_mark_plan_storage_demands,
            .user_data = NULL,
        },
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
