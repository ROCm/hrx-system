// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_rematerialization.h"

#include <string.h>

#include "loom/analysis/availability.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

static loom_low_allocation_rematerialization_result_t
loom_low_allocation_rematerialization_result_empty(void) {
  return (loom_low_allocation_rematerialization_result_t){
      .value_id = LOOM_VALUE_ID_INVALID,
      .assignment_index = UINT32_MAX,
  };
}

static bool loom_low_allocation_failure_is_rematerializable_pressure(
    const loom_low_allocation_failure_t* failure) {
  return loom_low_allocation_failure_is_present(failure) &&
         iree_string_view_equal(failure->failure_code,
                                IREE_SV("unspillable-register-exhausted"));
}

static bool loom_low_descriptor_packet_kind_may_rematerialize(
    loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ||
         kind == LOOM_LOW_DESCRIPTOR_PACKET_CONST;
}

static bool loom_low_allocation_rematerialization_use_is_eligible(
    loom_value_id_t value_id, const loom_op_t* defining_op, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  if (user_op == NULL || user_op == defining_op ||
      iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) ||
      user_op->parent_block == NULL ||
      operand_index >= user_op->operand_count) {
    return false;
  }
  if (loom_op_operands(user_op)[operand_index] != value_id) {
    return false;
  }
  if (user_op->parent_block == defining_op->parent_block &&
      user_op->block_ordinal <= defining_op->block_ordinal) {
    return false;
  }
  return true;
}

static bool loom_low_allocation_rematerialization_use_shortens_live_range(
    const loom_op_t* defining_op, loom_use_t use) {
  const loom_op_t* user_op = loom_use_user_op(use);
  // Cloning an already-adjacent producer cannot reduce pressure and can cycle.
  return user_op != NULL &&
         (user_op->parent_block != defining_op->parent_block ||
          user_op->prev_op != defining_op);
}

static iree_status_t loom_low_allocation_rematerialization_clone_for_use(
    loom_rewriter_t* rewriter, const loom_op_t* defining_op,
    uint16_t result_index, loom_value_id_t source_value_id, loom_use_t use,
    iree_arena_allocator_t* arena, loom_value_id_t* out_cloned_value_id) {
  *out_cloned_value_id = LOOM_VALUE_ID_INVALID;

  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(rewriter->module, rewriter->module, arena,
                               &(loom_ir_remap_options_t){
                                   .allow_unmapped_values = true,
                               },
                               &remap));

  loom_op_t* user_op = loom_use_user_op(use);
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, user_op);
  loom_op_t* cloned_op = NULL;
  iree_status_t status =
      loom_ir_clone_op(&rewriter->builder, defining_op, &remap, &cloned_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_ASSERT(result_index < cloned_op->result_count);
  const loom_value_id_t cloned_value_id =
      loom_op_results(cloned_op)[result_index];
  IREE_RETURN_IF_ERROR(
      loom_rewriter_clear_value_name(rewriter, cloned_value_id));
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      rewriter, source_value_id, cloned_value_id, IREE_SV("remat")));
  *out_cloned_value_id = cloned_value_id;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_try_rematerialize_value(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, const loom_use_t* requested_uses,
    uint32_t requested_use_count, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* result) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_is_consumed(value) ||
      loom_module_value_has_predicate_attribute_uses(module, value_id) ||
      loom_module_value_has_type_uses(module, value_id)) {
    return iree_ok_status();
  }

  const uint16_t result_index = loom_value_def_index(value);
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->result_count != 1 || defining_op->region_count != 0 ||
      defining_op->successor_count != 0 ||
      defining_op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_descriptor_packet(module, target, defining_op, &packet));
  if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
      !loom_low_descriptor_result_can_rematerialize(
          target->descriptor_set, packet.descriptor, result_index)) {
    return iree_ok_status();
  }

  const bool has_requested_uses = requested_uses != NULL;
  const loom_use_t* source_uses =
      has_requested_uses ? requested_uses : loom_value_uses(value);
  const uint32_t source_use_count =
      has_requested_uses ? requested_use_count : value->use_count;
  if (source_use_count == 0) return iree_ok_status();
  loom_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, source_use_count,
                                                 sizeof(*uses), (void**)&uses));
  uint32_t use_count = 0;
  bool shortens_live_range = false;
  for (uint32_t i = 0; i < source_use_count; ++i) {
    if (!loom_low_allocation_rematerialization_use_is_eligible(
            value_id, defining_op, source_uses[i])) {
      if (!has_requested_uses) return iree_ok_status();
      continue;
    }
    uses[use_count++] = source_uses[i];
    shortens_live_range =
        shortens_live_range ||
        loom_low_allocation_rematerialization_use_shortens_live_range(
            defining_op, source_uses[i]);
  }
  if (use_count == 0 || !shortens_live_range) return iree_ok_status();
  loom_availability_analysis_t availability = {0};
  IREE_RETURN_IF_ERROR(
      loom_availability_analysis_initialize(module, arena, &availability));
  uint32_t available_use_count = 0;
  for (uint32_t i = 0; i < use_count; ++i) {
    bool available = false;
    IREE_RETURN_IF_ERROR(loom_availability_op_captures_are_available_before_op(
        &availability, defining_op, loom_use_user_op(uses[i]), defining_op,
        &available));
    if (!available && !has_requested_uses) return iree_ok_status();
    if (available) uses[available_use_count++] = uses[i];
  }
  use_count = available_use_count;
  if (use_count == 0) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  iree_status_t status = loom_rewriter_initialize(&rewriter, module, arena);
  if (!iree_status_is_ok(status)) return status;
  for (uint32_t i = 0; i < use_count && iree_status_is_ok(status); ++i) {
    loom_value_id_t cloned_value_id = LOOM_VALUE_ID_INVALID;
    status = loom_low_allocation_rematerialization_clone_for_use(
        &rewriter, defining_op, result_index, value_id, uses[i], arena,
        &cloned_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operand(&rewriter, loom_use_user_op(uses[i]),
                                         loom_use_operand_index(uses[i]),
                                         cloned_value_id);
    }
    if (iree_status_is_ok(status)) {
      ++result->cloned_packet_count;
      ++result->rewritten_operand_count;
    }
  }
  if (iree_status_is_ok(status) &&
      loom_op_results_unused(module, defining_op)) {
    status = loom_rewriter_erase(&rewriter, defining_op);
  }
  if (iree_status_is_ok(status)) {
    result->value_id = value_id;
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  if (!loom_low_allocation_failure_is_rematerializable_pressure(
          &table->failure)) {
    return iree_ok_status();
  }

  const loom_low_allocation_failure_t* failure = &table->failure;
  if (failure->blocking_kind ==
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT &&
      failure->conflict_value_id != failure->value_id) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, failure->conflict_value_id,
        /*requested_uses=*/NULL, /*requested_use_count=*/0, arena, out_result));
    if (out_result->rewritten_operand_count != 0) {
      out_result->assignment_index = failure->conflict_assignment_index;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
      module, &table->target, failure->value_id,
      /*requested_uses=*/NULL, /*requested_use_count=*/0, arena, out_result));
  if (out_result->rewritten_operand_count != 0) {
    out_result->assignment_index = UINT32_MAX;
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_rematerialize_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, spill_plan->value_id,
        /*requested_uses=*/NULL, /*requested_use_count=*/0, arena, out_result));
    if (out_result->rewritten_operand_count != 0) {
      out_result->assignment_index = spill_plan->assignment_index;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

typedef struct loom_low_allocation_materialization_use_block_t {
  // Exact block receiving one reconstructed source materialization.
  loom_block_t* block;
  // Earliest recorded use in |block|.
  loom_op_t* before_op;
} loom_low_allocation_materialization_use_block_t;

static bool loom_low_allocation_materialization_value_is_ready(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_op_t* const* materialization_ops,
    uint32_t ready_materialization_op_count,
    const loom_value_id_t* materialization_inputs,
    uint32_t materialization_input_count) {
  for (uint32_t i = 0; i < materialization_input_count; ++i) {
    if (materialization_inputs[i] == value_id) return true;
  }
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* defining_op = loom_value_def_op(value);
  for (uint32_t i = 0; i < ready_materialization_op_count; ++i) {
    if (materialization_ops[i] == defining_op) return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_rematerialize_candidate_slice_uses(
    loom_module_t* module, loom_value_id_t value_id,
    loom_op_t* const* materialization_ops, uint32_t materialization_op_count,
    const loom_value_id_t* materialization_inputs,
    uint32_t materialization_input_count, const loom_use_t* requested_uses,
    uint32_t requested_use_count, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* result) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < materialization_op_count; ++i) {
    loom_op_t* op = materialization_ops[i];
    if (op == NULL || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
        op->parent_block == NULL || op->region_count != 0 ||
        op->successor_count != 0 || op->tied_result_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "recorded residency materialization slice became stale");
    }
    const loom_value_id_t* operands = loom_op_const_operands(op);
    for (uint16_t operand_index = 0; operand_index < op->operand_count;
         ++operand_index) {
      if (!loom_low_allocation_materialization_value_is_ready(
              module, operands[operand_index], materialization_ops, i,
              materialization_inputs, materialization_input_count)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "recorded residency materialization dependency became stale");
      }
    }
  }

  loom_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, requested_use_count,
                                                 sizeof(*uses), (void**)&uses));
  uint32_t use_count = 0;
  bool shortens_live_range = false;
  for (uint32_t i = 0; i < requested_use_count; ++i) {
    if (!loom_low_allocation_rematerialization_use_is_eligible(
            value_id, defining_op, requested_uses[i])) {
      continue;
    }
    uses[use_count++] = requested_uses[i];
    shortens_live_range =
        shortens_live_range ||
        loom_low_allocation_rematerialization_use_shortens_live_range(
            defining_op, requested_uses[i]);
  }
  if (use_count == 0 || !shortens_live_range) return iree_ok_status();

  loom_low_allocation_materialization_use_block_t* use_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, use_count, sizeof(*use_blocks), (void**)&use_blocks));
  uint32_t use_block_count = 0;
  for (uint32_t i = 0; i < use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    uint32_t block_index = 0;
    for (; block_index < use_block_count; ++block_index) {
      if (use_blocks[block_index].block == user_op->parent_block) break;
    }
    if (block_index == use_block_count) {
      use_blocks[use_block_count++] =
          (loom_low_allocation_materialization_use_block_t){
              .block = user_op->parent_block,
              .before_op = user_op,
          };
    } else if (user_op->block_ordinal <
               use_blocks[block_index].before_op->block_ordinal) {
      use_blocks[block_index].before_op = user_op;
    }
  }

  loom_availability_analysis_t availability = {0};
  IREE_RETURN_IF_ERROR(
      loom_availability_analysis_initialize(module, arena, &availability));
  for (uint32_t block_index = 0; block_index < use_block_count; ++block_index) {
    for (uint32_t input_index = 0; input_index < materialization_input_count;
         ++input_index) {
      const loom_value_id_t input = materialization_inputs[input_index];
      if (input == LOOM_VALUE_ID_INVALID || input >= module->values.count ||
          !loom_availability_value_is_available_before_op(
              &availability, /*moving_root_op=*/NULL,
              use_blocks[block_index].before_op, input)) {
        return iree_ok_status();
      }
    }
  }

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  iree_status_t status = iree_ok_status();
  for (uint32_t block_index = 0;
       block_index < use_block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_ir_remap_t remap;
    status = loom_ir_remap_initialize(
        module, module, arena,
        &(loom_ir_remap_options_t){.allow_unmapped_values = false}, &remap);
    for (uint32_t input_index = 0;
         input_index < materialization_input_count && iree_status_is_ok(status);
         ++input_index) {
      status =
          loom_ir_remap_map_value(&remap, materialization_inputs[input_index],
                                  materialization_inputs[input_index]);
    }
    if (!iree_status_is_ok(status)) break;

    loom_builder_set_before(&rewriter.builder,
                            use_blocks[block_index].before_op);
    for (uint32_t op_index = 0;
         op_index < materialization_op_count && iree_status_is_ok(status);
         ++op_index) {
      loom_op_t* cloned_op = NULL;
      status = loom_ir_clone_op(
          &rewriter.builder, materialization_ops[op_index], &remap, &cloned_op);
      if (iree_status_is_ok(status)) {
        if (result->cloned_packet_count == UINT32_MAX) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "residency materialization clone count exceeds uint32");
        } else {
          ++result->cloned_packet_count;
        }
      }
    }
    loom_value_id_t cloned_value_id = LOOM_VALUE_ID_INVALID;
    if (iree_status_is_ok(status)) {
      status = loom_ir_remap_resolve_value(&remap, value_id, &cloned_value_id);
    }
    for (uint32_t use_index = 0;
         use_index < use_count && iree_status_is_ok(status); ++use_index) {
      loom_op_t* user_op = loom_use_user_op(uses[use_index]);
      if (user_op->parent_block != use_blocks[block_index].block) continue;
      status = loom_rewriter_set_operand(
          &rewriter, user_op, loom_use_operand_index(uses[use_index]),
          cloned_value_id);
      if (iree_status_is_ok(status)) {
        if (result->rewritten_operand_count == UINT32_MAX) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "residency materialization rewrite count exceeds uint32");
        } else {
          ++result->rewritten_operand_count;
        }
      }
    }
  }
  for (uint32_t i = materialization_op_count;
       i > 0 && iree_status_is_ok(status); --i) {
    loom_op_t* op = materialization_ops[i - 1];
    if (loom_op_results_unused(module, op)) {
      status = loom_rewriter_erase(&rewriter, op);
    }
  }
  if (iree_status_is_ok(status) && result->rewritten_operand_count != 0) {
    result->value_id = value_id;
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_allocation_rematerialize_candidate_uses(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_value_id_t value_id, loom_op_t* const* materialization_ops,
    uint32_t materialization_op_count,
    const loom_value_id_t* materialization_inputs,
    uint32_t materialization_input_count, const loom_use_t* uses,
    uint32_t use_count, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  if (uses == NULL || use_count == 0) return iree_ok_status();
  if (materialization_op_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_candidate_slice_uses(
        module, value_id, materialization_ops, materialization_op_count,
        materialization_inputs, materialization_input_count, uses, use_count,
        arena, out_result));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, value_id, uses, use_count, arena, out_result));
  }
  if (out_result->rewritten_operand_count == 0) return iree_ok_status();
  for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
    if (table->assignments[i].value_id == value_id) {
      out_result->assignment_index = (uint32_t)i;
      break;
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_rematerialization_trigger_name(
    loom_low_allocation_rematerialization_trigger_t trigger) {
  switch (trigger) {
    case LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE:
      return IREE_SV("allocation-failure");
    case LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_SPILL_PLAN:
      return IREE_SV("spill-plan");
    case LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_RESIDENCY_CLIFF:
      return IREE_SV("residency-cliff");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t
loom_low_allocation_rematerialization_value_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_rematerialization_result_t* result) {
  if (result->assignment_index < table->assignment_count) {
    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[result->assignment_index];
    return loom_low_diagnostic_value_class_name(table->target.descriptor_set,
                                                assignment->value_class);
  }
  if (result->value_id == table->failure.value_id) {
    return loom_low_diagnostic_value_class_name(table->target.descriptor_set,
                                                table->failure.value_class);
  }
  return IREE_SV("<unknown>");
}

iree_status_t loom_low_allocation_rematerialization_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_rematerialization_trigger_t trigger,
    const loom_low_allocation_rematerialization_result_t* result,
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
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, result->value_id)),
      loom_param_string(loom_low_allocation_rematerialization_value_class_name(
          table, result)),
      loom_param_string(
          loom_low_allocation_rematerialization_trigger_name(trigger)),
      loom_param_u32(result->cloned_packet_count),
      loom_param_u32(result->rewritten_operand_count),
      loom_param_string(IREE_SV("descriptor-rematerializable")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = table->function_op,
      .error = LOOM_ERR_BACKEND_045,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
