// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_live_range_splitting.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

static loom_low_allocation_live_range_split_result_t
loom_low_allocation_live_range_split_result_empty(void) {
  return (loom_low_allocation_live_range_split_result_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
      .split_value_id = LOOM_VALUE_ID_INVALID,
      .source_assignment_index = UINT32_MAX,
  };
}

static bool loom_low_allocation_fixed_value_overlaps_spill_assignment(
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    const loom_low_allocation_assignment_t* spill_assignment) {
  return loom_liveness_value_class_equal(fixed_value->interval->value_class,
                                         spill_assignment->value_class) &&
         loom_low_allocation_live_range_assignment_overlaps_interval(
             spill_assignment, fixed_value->interval);
}

static bool loom_low_allocation_fixed_value_has_only_split_copy_use(
    const loom_value_t* value) {
  if (value->use_count != 1) {
    return false;
  }
  const loom_use_t use = loom_value_uses(value)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  return loom_low_copy_isa(user_op) && loom_use_operand_index(use) == 0;
}

static bool loom_low_allocation_split_use_is_eligible(
    loom_value_id_t value_id, const loom_block_t* insertion_block,
    const loom_op_t* insertion_anchor, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  if (user_op == NULL || iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) ||
      user_op->parent_block == NULL ||
      operand_index >= user_op->operand_count) {
    return false;
  }
  if (loom_op_operands(user_op)[operand_index] != value_id) {
    return false;
  }
  if (user_op->parent_block == insertion_block && insertion_anchor != NULL &&
      user_op->block_ordinal <= insertion_anchor->block_ordinal) {
    return false;
  }
  return true;
}

static bool loom_low_allocation_value_can_split_after_definition(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_block_t** out_insertion_block, loom_op_t** out_insertion_anchor) {
  *out_insertion_block = NULL;
  *out_insertion_anchor = NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    loom_block_t* defining_block = loom_value_def_block(value);
    if (defining_block == NULL) {
      return false;
    }
    *out_insertion_block = defining_block;
    return true;
  }
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->parent_block == NULL) {
    return false;
  }
  *out_insertion_block = defining_op->parent_block;
  *out_insertion_anchor = defining_op;
  return true;
}

static iree_status_t loom_low_allocation_try_split_fixed_value(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    uint32_t source_assignment_index, iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* result) {
  if (fixed_value->value_id == LOOM_VALUE_ID_INVALID ||
      fixed_value->value_id >= module->values.count) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, fixed_value->value_id);
  if (loom_value_is_consumed(value) ||
      loom_module_value_has_predicate_attribute_uses(module,
                                                     fixed_value->value_id) ||
      loom_module_value_has_type_uses(module, fixed_value->value_id) ||
      value->use_count == 0 ||
      loom_low_allocation_fixed_value_has_only_split_copy_use(value)) {
    return iree_ok_status();
  }

  loom_block_t* insertion_block = NULL;
  loom_op_t* insertion_anchor = NULL;
  if (!loom_low_allocation_value_can_split_after_definition(
          module, fixed_value->value_id, &insertion_block, &insertion_anchor)) {
    return iree_ok_status();
  }

  const uint32_t original_use_count = value->use_count;
  const loom_use_t* original_uses = loom_value_uses(value);
  for (uint32_t i = 0; i < original_use_count; ++i) {
    if (!loom_low_allocation_split_use_is_eligible(
            fixed_value->value_id, insertion_block, insertion_anchor,
            original_uses[i])) {
      return iree_ok_status();
    }
  }

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter.builder);
  if (insertion_anchor != NULL) {
    loom_builder_set_after(&rewriter.builder, insertion_anchor);
  } else if (insertion_block->first_op != NULL) {
    loom_builder_set_before(&rewriter.builder, insertion_block->first_op);
  } else {
    loom_builder_set_block(&rewriter.builder, insertion_block);
  }

  loom_op_t* copy_op = NULL;
  iree_status_t status =
      loom_low_copy_build(&rewriter.builder, fixed_value->value_id, true,
                          loom_module_value_type(module, fixed_value->value_id),
                          LOOM_LOCATION_NONE, &copy_op);
  loom_builder_restore(&rewriter.builder, saved_ip);
  if (iree_status_is_ok(status)) {
    const loom_value_id_t split_value_id = loom_low_copy_result(copy_op);
    status = loom_rewriter_try_set_derived_value_name(
        &rewriter, fixed_value->value_id, split_value_id, IREE_SV("split"));
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_replace_all_uses_except(
          &rewriter, fixed_value->value_id, split_value_id, copy_op);
    }
    if (iree_status_is_ok(status)) {
      result->source_value_id = fixed_value->value_id;
      result->split_value_id = split_value_id;
      result->source_assignment_index = source_assignment_index;
      result->copy_packet_count = 1;
      result->rewritten_operand_count = original_use_count;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_allocation_split_fixed_value_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* out_result) {
  *out_result = loom_low_allocation_live_range_split_result_empty();
  if (table->spill_plan_count == 0 || table->fixed_value_count == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    if (spill_plan->assignment_index >= table->assignment_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation spill plan references an invalid assignment");
    }
    const loom_low_allocation_assignment_t* spill_assignment =
        &table->assignments[spill_plan->assignment_index];
    for (iree_host_size_t j = 0; j < table->fixed_value_count; ++j) {
      const loom_low_allocation_resolved_fixed_value_t* fixed_value =
          &table->fixed_values[j];
      if (!loom_low_allocation_fixed_value_overlaps_spill_assignment(
              fixed_value, spill_assignment)) {
        continue;
      }
      uint32_t source_assignment_index = UINT32_MAX;
      loom_low_allocation_assignment_for_value_ordinal(
          table, fixed_value->value_ordinal, &source_assignment_index);
      IREE_RETURN_IF_ERROR(loom_low_allocation_try_split_fixed_value(
          module, table, fixed_value, source_assignment_index, arena,
          out_result));
      if (out_result->rewritten_operand_count != 0) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_live_range_split_trigger_name(
    loom_low_allocation_live_range_split_trigger_t trigger) {
  switch (trigger) {
    case LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN:
      return IREE_SV("spill-plan");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_live_range_split_value_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_live_range_split_result_t* result) {
  if (result->source_assignment_index < table->assignment_count) {
    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[result->source_assignment_index];
    return loom_low_diagnostic_value_class_name(table->target.descriptor_set,
                                                assignment->value_class);
  }
  for (iree_host_size_t i = 0; i < table->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &table->fixed_values[i];
    if (fixed_value->value_id == result->source_value_id) {
      return loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, fixed_value->interval->value_class);
    }
  }
  return IREE_SV("<unknown>");
}

iree_status_t loom_low_allocation_live_range_split_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_live_range_split_trigger_t trigger,
    const loom_low_allocation_live_range_split_result_t* result,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(result);
  if (emitter.fn == NULL || result->rewritten_operand_count == 0) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_value_name(
          table->module, result->source_value_id)),
      loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                       result->split_value_id)),
      loom_param_string(
          loom_low_allocation_live_range_split_value_class_name(table, result)),
      loom_param_string(
          loom_low_allocation_live_range_split_trigger_name(trigger)),
      loom_param_u32(result->copy_packet_count),
      loom_param_u32(result->rewritten_operand_count),
      loom_param_string(IREE_SV("fixed-value-spill-plan")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = table->function_op,
      .error = LOOM_ERR_BACKEND_046,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
