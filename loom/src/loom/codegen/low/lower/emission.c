// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/emission.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/codegen/low/lower/report.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/registers.h"

static iree_status_t loom_low_lower_emit_elided_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_elide_value(context, source_results[i]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_record_report_row(
      context, selected_plan, /*emitted_low_op_count=*/0));
  return iree_ok_status();
}

static loom_value_id_t loom_low_lower_descriptor_matrix_sparse_source_value(
    const loom_contract_request_t* request) {
  loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
  const loom_contract_operand_t* operands[] = {
      &request->lhs,
      &request->rhs,
      &request->accumulator,
      &request->result,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    const loom_contract_value_ref_t ref =
        operands[i]->encoded.auxiliary_value_refs
            [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA];
    if (!loom_contract_value_ref_is_present(ref)) {
      continue;
    }
    const loom_value_id_t operand_source_value =
        loom_contract_value_ref_value_id(ref);
    if (source_value == LOOM_VALUE_ID_INVALID) {
      source_value = operand_source_value;
      continue;
    }
    IREE_ASSERT_EQ(source_value, operand_source_value,
                   "descriptor-matrix selected sparse source is ambiguous");
  }
  IREE_ASSERT_NE(source_value, LOOM_VALUE_ID_INVALID,
                 "descriptor-matrix selected sparse source is unavailable");
  return source_value;
}

static loom_value_id_t loom_low_lower_descriptor_matrix_auxiliary_source_value(
    const loom_contract_operand_t* operand,
    loom_contract_auxiliary_operand_key_t key) {
  const loom_contract_value_ref_t ref =
      operand->encoded.auxiliary_value_refs[key];
  IREE_ASSERT(loom_contract_value_ref_is_present(ref),
              "descriptor-matrix selected auxiliary operand is unavailable");
  return loom_contract_value_ref_value_id(ref);
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_low_operand_source_binding_t source_binding, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t* out_low_value) {
  switch (source_binding) {
    case LOOM_LOW_OPERAND_SOURCE_BINDING_LHS:
      *out_low_value = low_lhs;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_RHS:
      *out_low_value = low_rhs;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_ACCUMULATOR:
      *out_low_value = low_init;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_SPARSE_METADATA: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_sparse_source_value(
              &plan->contract_request);
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_LHS_SCALE: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_auxiliary_source_value(
              &plan->contract_request.lhs,
              LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE);
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_RHS_SCALE: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_auxiliary_source_value(
              &plan->contract_request.rhs,
              LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE);
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_NONE:
    default:
      IREE_ASSERT_UNREACHABLE(
          "descriptor-matrix selected packet operand has no source binding");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t** out_operands, iree_host_size_t* out_operand_count) {
  *out_operands = NULL;
  *out_operand_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;

  iree_host_size_t operand_count = 0;
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const uint32_t row = descriptor->operand_start + i;
    const loom_low_operand_t* operand = &descriptor_set->operands[row];
    if (loom_low_operand_role_is_packet_operand(operand->role)) {
      ++operand_count;
    }
  }

  loom_value_id_t* operands = NULL;
  if (operand_count > 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, operand_count, sizeof(*operands), (void**)&operands));
  }

  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_value(
        context, plan, operand->source_binding, low_lhs, low_rhs, low_init,
        &operands[operand->source_value_index]));
  }

  *out_operands = operands;
  *out_operand_count = operand_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_tied_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  iree_host_size_t tied_result_count = 0;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t row = descriptor->constraint_start + i;
    if (descriptor_set->constraints[row].kind ==
        LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, tied_result_count, sizeof(*tied_results),
      (void**)&tied_results));
  iree_host_size_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    if (constraint->lhs_operand_index >= descriptor->result_count ||
        constraint->rhs_operand_index == LOOM_LOW_ID_NONE ||
        constraint->rhs_operand_index >= descriptor->operand_count) {
      IREE_ASSERT_UNREACHABLE(
          "descriptor-matrix selected tied result constraint is invalid");
      IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_operand_t* tied_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  constraint->rhs_operand_index];
    IREE_ASSERT_NE(tied_operand->source_value_index, LOOM_LOW_ID_NONE);
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = constraint->lhs_operand_index,
        .operand_index = tied_operand->source_value_index,
        .has_type_change = false,
    };
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static bool loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t constraint_index,
    uint16_t packet_operand_index) {
  for (uint16_t i = 0; i < constraint_index; ++i) {
    const loom_low_constraint_t* previous =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (previous->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(previous->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(previous->rhs_operand_index < descriptor->operand_count);
    const loom_low_operand_t* previous_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  previous->rhs_operand_index];
    if (previous_operand->source_value_index == packet_operand_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_descriptor_matrix_copy_destructive_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t* operands) {
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(constraint->lhs_operand_index < descriptor->result_count);
    IREE_ASSERT(constraint->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(constraint->rhs_operand_index < descriptor->operand_count);
    const loom_low_operand_t* destructive_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  constraint->rhs_operand_index];
    const uint16_t packet_operand_index =
        destructive_operand->source_value_index;
    IREE_ASSERT_NE(packet_operand_index, LOOM_LOW_ID_NONE);
    if (loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
            descriptor_set, descriptor, i, packet_operand_index)) {
      continue;
    }
    const loom_value_id_t source_value = operands[packet_operand_index];
    const loom_type_t copy_type =
        loom_module_value_type(context->module, source_value);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), source_value, false, copy_type,
        source_op->location, &copy_op));
    operands[packet_operand_index] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_vector_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  if (descriptor->result_count != 1) {
    IREE_ASSERT_UNREACHABLE(
        "descriptor-matrix vector.mma descriptor result count is invalid");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_lhs(source_op), &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_rhs(source_op), &low_rhs));
  loom_value_id_t low_init = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_init(source_op), &low_init));

  const loom_value_id_t result = loom_vector_mma_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, result, &result_low_type));
  IREE_ASSERT(loom_low_type_is_register(result_low_type));

  loom_value_id_t* operands = NULL;
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_operands(
      context, plan, low_lhs, low_rhs, low_init, &operands, &operand_count));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_descriptor_matrix_copy_destructive_operands(
          context, source_op, plan, operands));
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_tied_results(
      context, plan, &tied_results, &tied_result_count));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count, plan->attrs,
      &result_low_type, 1, tied_results, tied_result_count, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(
      context, result, loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  switch (plan->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA:
      return loom_low_lower_emit_descriptor_matrix_vector_mma(context,
                                                              source_op, plan);
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unknown descriptor-matrix source");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static uint64_t loom_low_lower_count_low_body_ops(
    const loom_low_lower_context_t* context) {
  uint64_t op_count = 0;
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  IREE_ASSERT(low_body != NULL);
  if (low_body == NULL) {
    return 0;
  }
  for (uint16_t block_index = 0; block_index < low_body->block_count;
       ++block_index) {
    op_count += loom_region_block(low_body, block_index)->op_count;
  }
  return op_count;
}

iree_status_t loom_low_lower_emit_selected_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_LT(context->lowering.selected_plan_emit_index,
                 context->lowering.selected_plan_count);
  const loom_low_lower_selected_plan_t selected_plan =
      context->lowering
          .selected_plans[context->lowering.selected_plan_emit_index++];
  IREE_ASSERT_EQ(selected_plan.source_op, source_op);
  if (iree_any_bit_set(selected_plan.flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED)) {
    return loom_low_lower_emit_elided_selected_plan(context, &selected_plan);
  }
  const bool report_allocator_provided =
      !iree_allocator_is_null(context->options->report_allocator);
  uint64_t before_op_count = 0;
  if (report_allocator_provided) {
    before_op_count = loom_low_lower_count_low_body_ops(context);
  }
  if (selected_plan.kind == LOOM_LOW_LOWER_SELECTED_PLAN_RULE) {
    IREE_ASSERT(selected_plan.rule_set != NULL);
    IREE_ASSERT(selected_plan.rule != NULL);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_emit_rule(
        context, selected_plan.rule_set, source_op, selected_plan.rule,
        selected_plan.resolved_emits, selected_plan.source_memory_access));
  } else if (selected_plan.kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_matrix_plan(
        context, source_op,
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan.plan.target_data));
  } else {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_ASSERT(context->policy->emit_op.fn != NULL);
    IREE_RETURN_IF_ERROR(
        context->policy->emit_op.fn(context->policy->emit_op.user_data, context,
                                    source_op, selected_plan.plan));
  }
  if (report_allocator_provided) {
    const uint64_t after_op_count = loom_low_lower_count_low_body_ops(context);
    IREE_ASSERT_GE(after_op_count, before_op_count);
    const uint64_t emitted_op_count = after_op_count - before_op_count;
    IREE_ASSERT_LE(emitted_op_count, UINT32_MAX);
    IREE_RETURN_IF_ERROR(loom_low_lower_record_report_row(
        context, &selected_plan, (uint32_t)emitted_op_count));
  }
  return iree_ok_status();
}
