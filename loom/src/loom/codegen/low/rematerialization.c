// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/rematerialization.h"

#include <string.h>

#include "loom/analysis/availability.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

void loom_low_rematerialization_invalidate_placement(
    loom_low_rematerialization_state_t* state) {
  iree_bitmap_reset_all(state->per_use_values);
}

static iree_status_t loom_low_rematerialization_reserve_per_use_values(
    loom_low_rematerialization_state_t* state,
    iree_host_size_t required_bit_count) {
  const iree_host_size_t required_word_count =
      iree_bitmap_calculate_words(required_bit_count);
  iree_host_size_t word_capacity =
      iree_bitmap_calculate_words(state->per_use_values.bit_count);
  if (required_word_count > word_capacity) {
    const iree_host_size_t old_word_count = word_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, old_word_count, required_word_count,
        sizeof(*state->per_use_values.words), &word_capacity,
        (void**)&state->per_use_values.words));
    memset(state->per_use_values.words + old_word_count, 0,
           (word_capacity - old_word_count) *
               sizeof(*state->per_use_values.words));
    state->per_use_values.bit_count = word_capacity * IREE_BITMAP_BITS_PER_WORD;
  }
  return iree_ok_status();
}

static loom_low_allocation_rematerialization_result_t
loom_low_allocation_rematerialization_result_empty(void) {
  return (loom_low_allocation_rematerialization_result_t){
      .value.value_id = LOOM_VALUE_ID_INVALID,
      .assignment_index = UINT32_MAX,
  };
}

static loom_low_value_rematerialization_result_t
loom_low_value_rematerialization_result_empty(void) {
  return (loom_low_value_rematerialization_result_t){
      .value_id = LOOM_VALUE_ID_INVALID,
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

static bool loom_low_rematerialization_use_is_eligible(
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

static bool loom_low_rematerialization_use_shortens_live_range(
    const loom_op_t* defining_op, loom_use_t use) {
  const loom_op_t* user_op = loom_use_user_op(use);
  // Cloning an already-adjacent producer cannot reduce pressure and can cycle.
  return user_op != NULL &&
         (user_op->parent_block != defining_op->parent_block ||
          user_op->prev_op != defining_op);
}

static iree_status_t loom_low_rematerialization_clone_for_use(
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

iree_status_t loom_low_rematerialize_value_uses(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, loom_low_rematerialization_state_t* state,
    iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result) {
  *out_result = loom_low_value_rematerialization_result_empty();
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (value_id < state->per_use_values.bit_count &&
      iree_bitmap_test(state->per_use_values, value_id)) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_is_consumed(value) ||
      loom_value_has_attribute_uses(value) || value->use_count == 0 ||
      loom_module_value_has_type_uses(module, value_id)) {
    return iree_ok_status();
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_low_register_type_resolver_t register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          target->descriptor_set);
  if (loom_low_register_type_resolver_has_class_flags(
          &register_type_resolver, value_type,
          LOOM_LOW_REG_CLASS_FLAG_REFERENCE)) {
    return iree_ok_status();
  }

  const uint16_t result_index = loom_value_def_index(value);
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->result_count != 1 || defining_op->region_count != 0 ||
      defining_op->successor_count != 0 ||
      defining_op->tied_result_count != 0 ||
      iree_any_bit_set(loom_op_effective_traits(module, defining_op),
                       LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return iree_ok_status();
  }

  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(target->descriptor_set, defining_op,
                                        &packet);
  if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
      !loom_low_descriptor_result_can_rematerialize(
          target->descriptor_set, packet.descriptor, result_index)) {
    return iree_ok_status();
  }

  const uint32_t use_count = value->use_count;
  loom_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, use_count,
                                                 sizeof(*uses), (void**)&uses));
  memcpy(uses, loom_value_uses(value), use_count * sizeof(*uses));
  bool shortens_live_range = false;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (!loom_low_rematerialization_use_is_eligible(value_id, defining_op,
                                                    uses[i])) {
      return iree_ok_status();
    }
    shortens_live_range = shortens_live_range ||
                          loom_low_rematerialization_use_shortens_live_range(
                              defining_op, uses[i]);
  }
  if (!shortens_live_range) {
    return iree_ok_status();
  }
  loom_availability_analysis_t availability = {0};
  IREE_RETURN_IF_ERROR(
      loom_availability_analysis_initialize(module, arena, &availability));
  for (uint32_t i = 0; i < use_count; ++i) {
    bool available = false;
    IREE_RETURN_IF_ERROR(loom_availability_op_captures_are_available_before_op(
        &availability, defining_op, loom_use_user_op(uses[i]), defining_op,
        &available));
    if (!available) {
      return iree_ok_status();
    }
  }

  // Each eligible packet has one result and no regions. Reserve membership for
  // its per-operand clones before any mutation changes the module value count.
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_per_use_values(
      state, module->values.count + use_count));
  loom_rewriter_t rewriter = {0};
  loom_low_value_rematerialization_result_t result = {
      .value_id = value_id,
  };
  loom_rewriter_initialize(&rewriter, module, arena);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < use_count && iree_status_is_ok(status); ++i) {
    loom_value_id_t cloned_value_id = LOOM_VALUE_ID_INVALID;
    status = loom_low_rematerialization_clone_for_use(
        &rewriter, defining_op, result_index, value_id, uses[i], arena,
        &cloned_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operand(&rewriter, loom_use_user_op(uses[i]),
                                         loom_use_operand_index(uses[i]),
                                         cloned_value_id);
    }
    if (iree_status_is_ok(status)) {
      iree_bitmap_set(state->per_use_values, cloned_value_id);
      ++result.cloned_packet_count;
      ++result.rewritten_operand_count;
    }
  }
  if (iree_status_is_ok(status)) {
    IREE_ASSERT(loom_op_results_unused(module, defining_op));
    status = loom_rewriter_erase(&rewriter, defining_op);
  }
  if (iree_status_is_ok(status)) {
    *out_result = result;
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_low_allocation_try_rematerialize_value(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, loom_low_rematerialization_state_t* state,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* result) {
  loom_low_value_rematerialization_result_t value_result = {0};
  IREE_RETURN_IF_ERROR(loom_low_rematerialize_value_uses(
      module, target, value_id, state, arena, &value_result));
  result->value = value_result;
  return iree_ok_status();
}

// Returns true when any storage unit in |assignment| is live at |point|.
//
// Whole-assignment bounds reject most candidates cheaply. Sparse storage
// segments exclude values that only appear live because mutually exclusive CFG
// paths share the linear program-point space, and refined per-unit bounds
// retain the allocation model's subrange precision.
static bool loom_low_allocation_assignment_is_live_at_point(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment, uint32_t point) {
  if (point < assignment->start_point || point >= assignment->end_point) {
    return false;
  }
  if (assignment->liveness_segments.count != 0 &&
      !loom_liveness_segment_range_contains(
          table->storage_segments, assignment->liveness_segments, point)) {
    return false;
  }
  for (uint32_t i = 0; i < assignment->unit_count; ++i) {
    const uint32_t start_point =
        loom_low_allocation_live_range_assignment_unit_start_point(
            table->unit_start_points, table->unit_point_count, assignment, i);
    const uint32_t end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            table->unit_end_points, table->unit_point_count, assignment, i);
    if (point >= start_point && point < end_point) {
      return true;
    }
  }
  return false;
}

typedef enum loom_low_allocation_rematerialization_frontier_e {
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_PRESSURE_CLASS = 0,
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_OVERLAPPING_STORAGE = 1,
} loom_low_allocation_rematerialization_frontier_t;

static bool loom_low_allocation_assignment_overlaps_failure_storage(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_low_allocation_failure_t* failure = &table->failure;
  const loom_low_descriptor_set_t* descriptor_set =
      table->target.descriptor_set;
  if (failure->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      failure->descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }

  const loom_low_reg_class_t* failed_reg_class =
      &descriptor_set->reg_classes[failure->descriptor_reg_class_id];
  loom_low_allocation_assignment_t candidate = {
      .value_id = failure->value_id,
      .value_class = failure->value_class,
      .descriptor_reg_class_id = failure->descriptor_reg_class_id,
      .unit_count = failure->required_unit_count,
      .location_kind = failure->location_kind,
      .location_count = failure->required_unit_count,
  };
  if (!loom_low_reg_class_uses_explicit_physical_registers(failed_reg_class)) {
    if (failure->location_base == UINT32_MAX || failure->location_count == 0) {
      return false;
    }
    candidate.location_base = failure->location_base;
    candidate.location_count = failure->location_count;
    return loom_low_allocation_storage_assignment_ranges_overlap(
        descriptor_set, &candidate, assignment);
  }

  for (uint32_t physical_register_id = 0;
       physical_register_id < descriptor_set->physical_register_count;
       ++physical_register_id) {
    if (!loom_low_allocation_storage_explicit_physical_register_view(
            descriptor_set, failure->descriptor_reg_class_id,
            physical_register_id, failure->required_unit_count,
            /*out_first_candidate_ordinal=*/NULL,
            /*out_pressure_extent=*/NULL)) {
      continue;
    }
    candidate.location_base = physical_register_id;
    if (loom_low_allocation_storage_assignment_ranges_overlap(
            descriptor_set, &candidate, assignment)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_rematerialization_frontier_contains(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    loom_low_allocation_rematerialization_frontier_t frontier) {
  const bool has_failed_class = loom_liveness_value_class_equal(
      assignment->value_class, table->failure.value_class);
  if (frontier ==
      LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_PRESSURE_CLASS) {
    return has_failed_class;
  }
  return !has_failed_class &&
         loom_low_allocation_assignment_overlaps_failure_storage(table,
                                                                 assignment);
}

// Attempts live values in one failed pressure frontier. Allocation reports one
// concrete collision, but that assignment is not necessarily the live value
// dominating the failure frontier. Candidates are ordered by remaining live
// unit-points so a successful rewrite removes the largest conservative
// pressure area first.
static iree_status_t loom_low_allocation_try_rematerialize_live_frontier(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_allocation_rematerialization_frontier_t frontier,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  const loom_low_allocation_failure_t* failure = &table->failure;
  uint64_t previous_pressure_area = UINT64_MAX;
  iree_host_size_t previous_assignment_index = IREE_HOST_SIZE_MAX;
  for (;;) {
    uint64_t best_pressure_area = 0;
    iree_host_size_t best_assignment_index = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
      const loom_low_allocation_assignment_t* assignment =
          &table->assignments[i];
      if (!loom_low_allocation_rematerialization_frontier_contains(
              table, assignment, frontier) ||
          !loom_low_allocation_assignment_is_live_at_point(
              table, assignment, failure->start_point)) {
        continue;
      }
      const uint64_t remaining_points =
          assignment->end_point - failure->start_point;
      uint64_t pressure_area = 0;
      if (!iree_checked_mul_u64(remaining_points, assignment->unit_count,
                                &pressure_area)) {
        pressure_area = UINT64_MAX;
      }
      if (previous_assignment_index != IREE_HOST_SIZE_MAX &&
          (pressure_area > previous_pressure_area ||
           (pressure_area == previous_pressure_area &&
            i <= previous_assignment_index))) {
        continue;
      }
      if (best_assignment_index == IREE_HOST_SIZE_MAX ||
          pressure_area > best_pressure_area ||
          (pressure_area == best_pressure_area && i < best_assignment_index)) {
        best_pressure_area = pressure_area;
        best_assignment_index = i;
      }
    }
    if (best_assignment_index == IREE_HOST_SIZE_MAX) {
      return iree_ok_status();
    }

    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[best_assignment_index];
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, assignment->value_id, state, arena,
        out_result));
    if (out_result->value.rewritten_operand_count != 0) {
      out_result->assignment_index = (uint32_t)best_assignment_index;
      return iree_ok_status();
    }
    previous_pressure_area = best_pressure_area;
    previous_assignment_index = best_assignment_index;
  }
}

iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  if (!loom_low_allocation_failure_is_rematerializable_pressure(
          &table->failure)) {
    return iree_ok_status();
  }

  const loom_low_allocation_failure_t* failure = &table->failure;
  IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_live_frontier(
      module, table,
      LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_PRESSURE_CLASS, state,
      arena, out_result));
  if (out_result->value.rewritten_operand_count != 0) {
    return iree_ok_status();
  }

  // Explicit physical classes can overlap several other register classes and
  // locations. Exhaust the whole allocatable storage domain instead of only
  // the first concrete collision selected for diagnostics.
  IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_live_frontier(
      module, table,
      LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_OVERLAPPING_STORAGE, state,
      arena, out_result));
  if (out_result->value.rewritten_operand_count != 0) {
    return iree_ok_status();
  }

  // The failed value may not have an assignment in the partial table and is
  // therefore the only pressure candidate not covered by the live frontier.
  IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
      module, &table->target, failure->value_id, state, arena, out_result));
  if (out_result->value.rewritten_operand_count != 0) {
    out_result->assignment_index = UINT32_MAX;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_rematerialize_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, spill_plan->value_id, state, arena,
        out_result));
    if (out_result->value.rewritten_operand_count != 0) {
      out_result->assignment_index = spill_plan->assignment_index;
      return iree_ok_status();
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
  if (result->value.value_id == table->failure.value_id) {
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
  if (emitter.fn == NULL || result->value.rewritten_operand_count == 0) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                       result->value.value_id)),
      loom_param_string(loom_low_allocation_rematerialization_value_class_name(
          table, result)),
      loom_param_string(
          loom_low_allocation_rematerialization_trigger_name(trigger)),
      loom_param_u32(result->value.cloned_packet_count),
      loom_param_u32(result->value.rewritten_operand_count),
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
