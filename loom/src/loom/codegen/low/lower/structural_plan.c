// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/structural_plan.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/source_value_storage.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/ref/ops.h"
#include "loom/ops/scf/ops.h"

static iree_status_t loom_low_lower_structural_require_blocks(
    loom_low_lower_context_t* context) {
  loom_low_lower_structural_plan_t* plan =
      &context->lowering->source_plan.structural;
  if (plan->branch_conditions != NULL) {
    return iree_ok_status();
  }
  const uint16_t count =
      loom_func_like_body(context->source_function)->block_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, count, sizeof(*plan->branch_conditions),
      (void**)&plan->branch_conditions));
  memset(plan->branch_conditions, 0, count * sizeof(*plan->branch_conditions));
  return iree_ok_status();
}

iree_status_t loom_low_lower_structural_plan_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      context->fact_table, loom_cfg_cond_br_condition(source_op));
  bool condition = false;
  if (loom_value_facts_as_exact_bool(facts, &condition)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_structural_require_blocks(context));
    context->lowering->source_plan.structural
        .branch_conditions[source_op->parent_block->region_index] =
        condition ? 2 : 1;
  }
  return iree_ok_status();
}

bool loom_low_lower_structural_branch_exact_bool(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition) {
  const uint8_t* conditions =
      context->lowering->source_plan.structural.branch_conditions;
  const uint8_t condition =
      conditions ? conditions[source_op->parent_block->region_index] : 0;
  if (out_condition != NULL) {
    *out_condition = condition == 2;
  }
  return condition != 0;
}

static iree_status_t loom_low_lower_structural_plan_types(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* source_values, uint16_t count,
    bool retain_bindings) {
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < count && iree_status_is_ok(status) &&
                       context->result->error_count == 0;
       ++i) {
    loom_type_t type = loom_type_none();
    status =
        loom_low_lower_map_value(context, source_op, source_values[i], &type);
    if (iree_status_is_ok(status) && retain_bindings &&
        context->result->error_count == 0) {
      status = loom_low_lower_plan_value_type(context, source_values[i], type);
    }
  }
  return status;
}

iree_status_t loom_low_lower_structural_plan_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_block) {
  return loom_low_lower_structural_plan_types(
      context, source_op, source_block->arg_ids, source_block->arg_count,
      source_block->parent_region ==
          loom_func_like_body(context->source_function));
}

loom_type_t loom_low_lower_structural_block_argument_type(
    const loom_low_lower_context_t* context, uint16_t block_index,
    uint16_t argument_index) {
  const loom_block_t* block = loom_region_const_block(
      loom_func_like_body(context->source_function), block_index);
  return loom_low_lower_value_binding_type(context,
                                           block->arg_ids[argument_index]);
}

iree_status_t loom_low_lower_structural_plan_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits) ||
      loom_traits_are_value_alias(traits)) {
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      loom_low_lower_inherit_value_type(context,
                                        loom_op_const_operands(source_op)[i],
                                        loom_op_const_results(source_op)[i]);
    }
    return iree_ok_status();
  }
  if (loom_low_lower_source_value_storage_op_is_duplication(context, source_op,
                                                            traits)) {
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      loom_low_lower_inherit_value_type(context,
                                        loom_op_const_operands(source_op)[i],
                                        loom_op_const_results(source_op)[i]);
    }
    return iree_ok_status();
  }
  if (loom_ref_borrow_isa(source_op)) {
    return loom_low_lower_structural_plan_types(
        context, source_op, loom_op_const_results(source_op),
        source_op->result_count, /*retain_bindings=*/true);
  }
  if (loom_buffer_assume_same_root_isa(source_op)) {
    loom_low_lower_inherit_value_type(
        context, loom_buffer_assume_same_root_buffer(source_op),
        loom_buffer_assume_same_root_result(source_op));
    return iree_ok_status();
  }
  if (loom_scf_for_isa(source_op) &&
      loom_scf_for_pipeline_depth_is_present(source_op)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("pipeline")),
        loom_param_i64(0),
        loom_param_string(
            IREE_SV("consumed by pipeline-scf-for before source-to-low")),
    };
    return loom_low_lower_emit_target_context_error(
        context, source_op, LOOM_ERR_STRUCTURE_014, params,
        IREE_ARRAYSIZE(params));
  }
  if (loom_scf_for_isa(source_op)) {
    const loom_block_t* body =
        loom_region_const_entry_block(loom_scf_for_body(source_op));
    loom_low_lower_inherit_value_type(
        context, loom_scf_for_lower_bound(source_op), body->arg_ids[0]);
    const loom_value_slice_t iter_args = loom_scf_for_iter_args(source_op);
    for (uint16_t i = 0; i < iter_args.count; ++i) {
      loom_low_lower_inherit_value_type(context, iter_args.values[i],
                                        body->arg_ids[i + 1]);
      loom_low_lower_inherit_value_type(context, iter_args.values[i],
                                        loom_op_const_results(source_op)[i]);
    }
    return iree_ok_status();
  }
  const loom_block_t* header =
      loom_scf_while_isa(source_op)
          ? loom_region_const_entry_block(loom_scf_while_before(source_op))
          : NULL;
  const uint16_t header_count = header ? header->arg_count : 0;
  if (!loom_scf_if_isa(source_op) && header == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_structural_plan_types(
      context, source_op, loom_op_const_results(source_op),
      source_op->result_count, /*retain_bindings=*/true));
  if (header != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_structural_plan_types(
        context, source_op, header->arg_ids, header_count,
        /*retain_bindings=*/true));
    const loom_block_t* after =
        loom_region_const_entry_block(loom_scf_while_after(source_op));
    for (uint16_t i = 0; i < after->arg_count; ++i) {
      loom_low_lower_inherit_value_type(
          context, loom_op_const_results(source_op)[i], after->arg_ids[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_structural_expand_types(
    loom_low_lower_context_t* context, const loom_value_id_t* source_values,
    uint16_t count, loom_type_t** out_types) {
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, count, sizeof(**out_types), (void**)out_types));
  for (uint16_t i = 0; i < count; ++i) {
    (*out_types)[i] =
        loom_low_lower_value_binding_type(context, source_values[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_structural_take_types(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t** out_result_types, loom_type_t** out_header_types) {
  *out_result_types = NULL;
  const uint16_t header_count =
      loom_scf_while_isa(source_op)
          ? loom_region_const_entry_block(loom_scf_while_before(source_op))
                ->arg_count
          : 0;
  if (out_header_types != NULL) {
    *out_header_types = NULL;
  }
  if (source_op->result_count == 0 && header_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_structural_expand_types(
      context, loom_op_const_results(source_op), source_op->result_count,
      out_result_types));
  if (header_count != 0) {
    const loom_block_t* header =
        loom_region_const_entry_block(loom_scf_while_before(source_op));
    IREE_RETURN_IF_ERROR(loom_low_lower_structural_expand_types(
        context, header->arg_ids, header_count, out_header_types));
  }
  return iree_ok_status();
}
