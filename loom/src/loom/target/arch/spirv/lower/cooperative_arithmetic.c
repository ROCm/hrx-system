// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/cooperative_arithmetic.h"

#include "loom/codegen/low/lower/participation.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/lower/matrix.h"
#include "loom/target/arch/spirv/ops/types.h"

typedef struct loom_spirv_cooperative_arithmetic_row_t {
  // Native matrix type, independent of the source vector's opaque lane count.
  loom_spirv_value_type_t type;
  // Descriptor ordinals plus one for native arithmetic and component math;
  // zero denotes an operation incompatible with this component type.
  uint32_t descriptors[25];
} loom_spirv_cooperative_arithmetic_row_t;

#include "loom/target/arch/spirv/lower/cooperative_arithmetic_rows.inl"

enum { LOOM_SPIRV_COOPERATIVE_ARITHMETIC_PLAN = 0x434d };

static uint8_t loom_spirv_cooperative_arithmetic_slot(const loom_op_t* op) {
  const uint32_t index = (uint32_t)op->kind - LOOM_OP_VECTOR_ADDF;
  return index < IREE_ARRAYSIZE(kCooperativeArithmeticSlots)
             ? kCooperativeArithmeticSlots[index]
             : 0;
}

static const loom_spirv_cooperative_arithmetic_row_t*
loom_spirv_cooperative_arithmetic_match(const loom_module_t* module,
                                        const loom_value_fact_table_t* facts,
                                        const loom_op_t* op, bool* out_native) {
  *out_native = false;
  loom_spirv_value_type_t type = {0};
  bool compatible = true;
  for (uint16_t i = 0; i <= op->operand_count; ++i) {
    const loom_value_id_t value = i == op->operand_count
                                      ? loom_op_const_results(op)[0]
                                      : loom_op_const_operands(op)[i];
    loom_vector_fragment_fact_t fragment;
    loom_spirv_value_type_t operand_type;
    const bool native = loom_spirv_matrix_fragment(facts, value, &fragment);
    *out_native |= native;
    if (!native || !loom_spirv_matrix_fragment_type(module, facts, value,
                                                    fragment, &operand_type)) {
      compatible = false;
    } else if (i == 0) {
      type = operand_type;
    } else if (!loom_spirv_value_type_equal(type, operand_type)) {
      compatible = false;
    }
  }
  if (!compatible) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCooperativeArithmeticRows);
       ++i) {
    if (loom_spirv_value_type_equal(type, kCooperativeArithmeticRows[i].type)) {
      return &kCooperativeArithmeticRows[i];
    }
  }
  return NULL;
}

iree_status_t loom_spirv_query_cooperative_arithmetic(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  const uint8_t slot = loom_spirv_cooperative_arithmetic_slot(source_op);
  if (slot == 0) {
    return iree_ok_status();
  }
  bool native = false;
  const loom_spirv_cooperative_arithmetic_row_t* row =
      loom_spirv_cooperative_arithmetic_match(
          environment->module, environment->fact_table, source_op, &native);
  if (!native) {
    return iree_ok_status();
  }
  if (environment->vector_lane_projection.source_lane_count != 0) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
    return iree_ok_status();
  }
  if (!iree_all_bits_set(source_op->instance_flags,
                         kCooperativeArithmeticRequiredFlags[slot - 1])) {
    IREE_RETURN_IF_ERROR(loom_spirv_matrix_contract_query_reject(
        environment, source_op,
        IREE_SV("cooperative_matrix.arithmetic_permissions"), 0, 0, 0,
        out_result));
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_TERMINAL_REJECTION;
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor =
      row != NULL && row->descriptors[slot - 1] != 0
          ? loom_low_descriptor_set_descriptor_at(
                environment->descriptor_set, row->descriptors[slot - 1] - 1)
          : NULL;
  uint64_t missing_features = 0;
  if (descriptor != NULL) {
    const loom_low_descriptor_layout_t* layout =
        loom_low_descriptor_layout(environment->descriptor_set, descriptor);
    const uint64_t required_features =
        environment->descriptor_set
            ->feature_mask_words[layout->feature_mask_word_start];
    missing_features =
        required_features &
        ~loom_target_contract_query_environment_bundle(environment)
             ->config->contract_feature_bits;
  }
  if (descriptor == NULL || missing_features != 0) {
    IREE_RETURN_IF_ERROR(loom_spirv_matrix_contract_query_reject(
        environment, source_op,
        missing_features ? IREE_SV("cooperative_matrix.arithmetic_features")
                         : IREE_SV("cooperative_matrix.arithmetic_types"),
        0, 0, missing_features, out_result));
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_TERMINAL_REJECTION;
    return iree_ok_status();
  }
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor = descriptor;
  return iree_ok_status();
}

bool loom_spirv_cooperative_arithmetic_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_SPIRV_COOPERATIVE_ARITHMETIC_PLAN;
}

iree_status_t loom_spirv_select_cooperative_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const uint8_t slot = loom_spirv_cooperative_arithmetic_slot(source_op);
  if (slot == 0) {
    return iree_ok_status();
  }
  bool native = false;
  const loom_spirv_cooperative_arithmetic_row_t* row =
      loom_spirv_cooperative_arithmetic_match(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op, &native);
  if (row == NULL || row->descriptors[slot - 1] == 0) {
    return iree_ok_status();
  }

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context),
          row->descriptors[slot - 1] - 1);
  *out_plan = loom_low_lower_plan_make(LOOM_SPIRV_COOPERATIVE_ARITHMETIC_PLAN,
                                       descriptor);
  loom_low_lower_participation_t participation;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_subgroup_participation(
      context, source_op, &participation));
  if (participation.kind != LOOM_LOW_LOWER_PARTICIPATION_FULL) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_lower_context_target_key(context)),
        loom_param_string(loom_low_lower_context_export_name(context)),
        loom_param_string(loom_low_lower_context_config_key(context)),
        loom_param_string(loom_low_lower_context_function_name(context)),
        loom_param_string(
            loom_op_name(loom_low_lower_context_module(context), source_op)),
        loom_param_string(IREE_SV("spirv.cooperative_matrix")),
        loom_param_string(IREE_SV("full_subgroup_participation")),
    };
    return loom_low_lower_emit_error_ref(context, source_op,
                                         LOOM_ERR_TARGET_039_REF, params,
                                         IREE_ARRAYSIZE(params));
  }
  loom_type_t matrix_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_spirv_cooperative_matrix_type_make(
      loom_low_lower_context_module(context), row->type.cooperative_matrix.rows,
      row->type.cooperative_matrix.columns, row->type.scalar_type,
      row->type.cooperative_matrix.scope, row->type.cooperative_matrix.use,
      &matrix_type));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1, matrix_type,
      &result_type));
  return loom_low_lower_plan_value_type(
      context, loom_op_const_results(source_op)[0], result_type);
}

iree_status_t loom_spirv_emit_cooperative_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = plan.target_data};
  loom_value_id_t operands[2];
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    operands[i] = loom_low_lower_lookup_value(
        context, loom_op_const_operands(source_op)[i]);
  }
  loom_named_attr_t contraction;
  loom_named_attr_slice_t attributes = loom_named_attr_slice_empty();
  if (loom_low_descriptor_layout(loom_low_lower_context_descriptor_set(context),
                                 descriptor.descriptor)
          ->immediate_count != 0) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context), IREE_SV("allow_contraction"),
        &contraction.name_id));
    contraction.value = loom_attr_i64(
        (source_op->instance_flags & LOOM_VECTOR_FASTMATHFLAGS_CONTRACT) != 0);
    attributes = loom_make_named_attr_slice(&contraction, 1);
  }
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_type_t type = loom_low_lower_value_binding_type(context, result);
  loom_op_t* emitted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, source_op->operand_count, attributes,
      &type, 1, NULL, 0, source_op->location, &emitted));
  return loom_low_lower_bind_value(context, result,
                                   loom_op_const_results(emitted)[0]);
}
