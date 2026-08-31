// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame.h"

#include <string.h>

#include "loom/codegen/low/addressability.h"
#include "loom/codegen/low/allocation_live_range_splitting.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/rematerialization.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/low/ops.h"

typedef enum loom_low_emission_frame_failure_e {
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS = 0,
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT = 1,
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS = 2,
} loom_low_emission_frame_failure_t;

// Value repairs rebuild whole-function scheduling and allocation. Every repair
// callback operates on a complete snapshot, so reaching this limit indicates a
// non-convergent value strategy rather than useful fine-grained progress.
#define LOOM_LOW_EMISSION_FRAME_MAX_VALUE_REPAIR_ITERATIONS 8

// Spill materialization may expose new spill traffic that requires another
// complete frame build. Bound that convergence process independently from
// value repair so one strategy cannot consume the other's budget.
#define LOOM_LOW_EMISSION_FRAME_MAX_SPILL_MATERIALIZATION_ITERATIONS 8

typedef struct loom_low_emission_frame_materialization_summary_t {
  // Cumulative spill storage materialized while building the final frame.
  uint64_t spill_storage_count;
  // Cumulative materialized spill storage byte size.
  uint64_t spill_storage_bytes;
  // Cumulative low.spill stores materialized while building the final frame.
  uint64_t spill_store_count;
  // Cumulative materialized low.spill store byte traffic.
  uint64_t spill_store_bytes;
  // Cumulative low.reload ops materialized while building the final frame.
  uint64_t reload_count;
  // Cumulative materialized low.reload byte traffic.
  uint64_t reload_bytes;
  // Cumulative materialized spill records retained for report detail rows.
  loom_low_allocation_materialized_spill_list_t spill_records;
} loom_low_emission_frame_materialization_summary_t;

static iree_status_t loom_low_emission_frame_reserve_value_bitmap(
    iree_host_size_t required_bit_count, iree_arena_allocator_t* arena,
    iree_bitmap_t* inout_values) {
  const iree_host_size_t required_word_count =
      iree_bitmap_calculate_words(required_bit_count);
  iree_host_size_t word_capacity =
      iree_bitmap_calculate_words(inout_values->bit_count);
  if (required_word_count > word_capacity) {
    const iree_host_size_t old_word_count = word_capacity;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, old_word_count, required_word_count,
                              sizeof(*inout_values->words), &word_capacity,
                              (void**)&inout_values->words));
    memset(inout_values->words + old_word_count, 0,
           (word_capacity - old_word_count) * sizeof(*inout_values->words));
    inout_values->bit_count = word_capacity * IREE_BITMAP_BITS_PER_WORD;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_record_value_span(
    iree_host_size_t first_value_id, iree_host_size_t end_value_id,
    iree_arena_allocator_t* arena, iree_bitmap_t* inout_values) {
  if (first_value_id == end_value_id) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_reserve_value_bitmap(
      end_value_id, arena, inout_values));
  iree_bitmap_set_span(*inout_values, first_value_id,
                       end_value_id - first_value_id);
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_record_value_list(
    const loom_value_id_t* value_ids, iree_host_size_t value_count,
    iree_arena_allocator_t* arena, iree_bitmap_t* inout_values) {
  iree_host_size_t required_bit_count = 0;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    required_bit_count =
        iree_max(required_bit_count, (iree_host_size_t)value_ids[i] + 1);
  }
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_reserve_value_bitmap(
      required_bit_count, arena, inout_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    iree_bitmap_set(*inout_values, value_ids[i]);
  }
  return iree_ok_status();
}

static bool loom_low_emission_frame_value_requires_register(
    iree_bitmap_t required_register_values, loom_value_id_t value_id) {
  return value_id < required_register_values.bit_count &&
         iree_bitmap_test(required_register_values, value_id);
}

static iree_status_t
loom_low_emission_frame_record_derived_register_requirement(
    loom_value_id_t source_value_id, iree_host_size_t first_derived_value_id,
    iree_host_size_t end_derived_value_id, iree_arena_allocator_t* repair_arena,
    iree_bitmap_t* inout_required_register_values) {
  if (!loom_low_emission_frame_value_requires_register(
          *inout_required_register_values, source_value_id)) {
    return iree_ok_status();
  }
  return loom_low_emission_frame_record_value_span(
      first_derived_value_id, end_derived_value_id, repair_arena,
      inout_required_register_values);
}

static void loom_low_emission_frame_record_arena_high_water(
    const iree_arena_allocator_t* arena, iree_host_size_t baseline_used_bytes,
    iree_host_size_t baseline_owned_bytes,
    loom_low_planning_arena_statistics_t* statistics) {
  IREE_ASSERT_GE(arena->used_allocation_size, baseline_used_bytes);
  IREE_ASSERT_GE(arena->total_allocation_size, baseline_owned_bytes);
  if (arena->used_allocation_size < baseline_used_bytes ||
      arena->total_allocation_size < baseline_owned_bytes) {
    return;
  }
  statistics->used_bytes_high_water =
      iree_max(statistics->used_bytes_high_water,
               (uint64_t)(arena->used_allocation_size - baseline_used_bytes));
  statistics->owned_bytes_high_water =
      iree_max(statistics->owned_bytes_high_water,
               (uint64_t)(arena->total_allocation_size - baseline_owned_bytes));
}

static void loom_low_emission_frame_record_memory_high_water(
    const iree_arena_checkpoint_t* frame_checkpoint,
    const iree_arena_allocator_t* repair_arena,
    const iree_arena_allocator_t* scratch_arena,
    loom_low_planning_statistics_t* statistics) {
  if (statistics == NULL) return;
  loom_low_emission_frame_record_arena_high_water(
      frame_checkpoint->arena, frame_checkpoint->used_allocation_size,
      frame_checkpoint->total_allocation_size, &statistics->memory.frame_arena);
  loom_low_emission_frame_record_arena_high_water(
      repair_arena, /*baseline_used_bytes=*/0, /*baseline_owned_bytes=*/0,
      &statistics->memory.repair_arena);
  loom_low_emission_frame_record_arena_high_water(
      scratch_arena, /*baseline_used_bytes=*/0, /*baseline_owned_bytes=*/0,
      &statistics->memory.scratch_arena);
}

#if IREE_STATISTICS_ENABLE
static void loom_low_emission_frame_record_system_allocation_delta(
    const iree_arena_block_pool_t* block_pool,
    const iree_arena_block_pool_statistics_t* before,
    loom_low_planning_statistics_t* statistics) {
  iree_arena_block_pool_statistics_t after = {0};
  iree_arena_block_pool_query_statistics(block_pool, &after);
  IREE_ASSERT_GE(after.block_system_allocation_count,
                 before->block_system_allocation_count);
  IREE_ASSERT_GE(after.block_system_allocation_bytes,
                 before->block_system_allocation_bytes);
  IREE_ASSERT_GE(after.oversized_allocation_count,
                 before->oversized_allocation_count);
  IREE_ASSERT_GE(after.oversized_allocation_bytes,
                 before->oversized_allocation_bytes);
  if (after.block_system_allocation_count <
          before->block_system_allocation_count ||
      after.block_system_allocation_bytes <
          before->block_system_allocation_bytes ||
      after.oversized_allocation_count < before->oversized_allocation_count ||
      after.oversized_allocation_bytes < before->oversized_allocation_bytes) {
    return;
  }
  statistics->flags |= LOOM_LOW_PLANNING_STATISTICS_FLAG_SYSTEM_ALLOCATIONS;
  statistics->memory.block_system_allocation_count =
      after.block_system_allocation_count -
      before->block_system_allocation_count;
  statistics->memory.block_system_allocation_bytes =
      after.block_system_allocation_bytes -
      before->block_system_allocation_bytes;
  statistics->memory.oversized_allocation_count =
      after.oversized_allocation_count - before->oversized_allocation_count;
  statistics->memory.oversized_allocation_bytes =
      after.oversized_allocation_bytes - before->oversized_allocation_bytes;
}
#endif  // IREE_STATISTICS_ENABLE

static void loom_low_emission_frame_advance_repair_iteration(
    iree_host_size_t* iteration_count,
    loom_low_planning_statistics_t* statistics) {
  ++*iteration_count;
  if (statistics != NULL) ++statistics->repair.iteration_count;
}

static iree_status_t loom_low_emission_frame_build_with_diagnostic_emitter(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    loom_low_placement_pair_use_list_t preferred_pair_uses,
    iree_bitmap_t required_register_values,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_low_planning_statistics_t* statistics,
    loom_low_emission_frame_t* out_frame) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }

  *out_frame = (loom_low_emission_frame_t){
      .module = module,
      .function_op = low_func_op,
  };
  if (statistics != NULL) ++statistics->frame_build_count;

  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      module, low_func_op, options->function_target_facts,
      options->descriptor_registry, diagnostic_emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE, arena, &model);
  loom_low_schedule_options_t schedule_options = {
      .memory_access_table = options->memory_access_table,
      .residency_model = options->residency_model,
      .allocation_budgets = options->allocation_budgets,
      .allocation_budget_count = options->allocation_budget_count,
      .pair_affinities = options->schedule_pair_affinities,
      .preferred_pair_uses = preferred_pair_uses,
      .structural_state_reads = options->schedule_structural_state_reads,
      .structural_models = options->schedule_structural_models,
      .emitter = diagnostic_emitter,
      .diagnostic_flags = options->schedule_diagnostic_flags,
      .strategy = options->schedule_strategy,
  };
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_function(&model, &schedule_options, arena,
                                        &out_frame->schedule);
  }

  loom_low_storage_lease_table_t storage_leases = {0};
  if (iree_status_is_ok(status) && out_frame->schedule.error_count == 0 &&
      options->storage_lease_provider != NULL) {
    status = loom_low_storage_lease_build(&out_frame->schedule,
                                          options->storage_lease_provider,
                                          arena, &storage_leases);
  }

  loom_low_allocation_options_t allocation_options = {
      .schedule = &out_frame->schedule,
      .budgets = options->allocation_budgets,
      .budget_count = options->allocation_budget_count,
      .fixed_values = options->allocation_fixed_values,
      .fixed_value_count = options->allocation_fixed_value_count,
      .reserved_ranges = options->allocation_reserved_ranges,
      .reserved_range_count = options->allocation_reserved_range_count,
      .required_register_values = required_register_values,
      .residency_model = options->residency_model,
      .storage_leases = storage_leases,
      .emitter = diagnostic_emitter,
      .diagnostic_flags = options->allocation_diagnostic_flags,
  };
  if (iree_status_is_ok(status) && out_frame->schedule.error_count == 0) {
    if (statistics != NULL) ++statistics->allocation_run_count;
    status = loom_low_allocate_function(&model, &allocation_options, arena,
                                        &out_frame->allocation);
  }
  if (iree_status_is_ok(status)) {
    out_frame->target = out_frame->schedule.target;
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

iree_status_t loom_low_emission_frame_build(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  loom_low_planning_statistics_t* statistics = options->statistics;
  if (statistics == NULL) {
    return loom_low_emission_frame_build_with_diagnostic_emitter(
        module, low_func_op, options, loom_low_placement_pair_use_list_empty(),
        (iree_bitmap_t){0}, options->emitter, arena,
        /*statistics=*/NULL, out_frame);
  }
  *statistics = (loom_low_planning_statistics_t){0};
  const iree_arena_checkpoint_t frame_checkpoint =
      iree_arena_checkpoint_save(arena);
#if IREE_STATISTICS_ENABLE
  iree_arena_block_pool_statistics_t pool_statistics_before = {0};
  iree_arena_block_pool_query_statistics(arena->block_pool,
                                         &pool_statistics_before);
#endif  // IREE_STATISTICS_ENABLE
  iree_status_t status = loom_low_emission_frame_build_with_diagnostic_emitter(
      module, low_func_op, options, loom_low_placement_pair_use_list_empty(),
      (iree_bitmap_t){0}, options->emitter, arena, statistics, out_frame);
  loom_low_emission_frame_record_arena_high_water(
      arena, frame_checkpoint.used_allocation_size,
      frame_checkpoint.total_allocation_size, &statistics->memory.frame_arena);
#if IREE_STATISTICS_ENABLE
  loom_low_emission_frame_record_system_allocation_delta(
      arena->block_pool, &pool_statistics_before, statistics);
#endif  // IREE_STATISTICS_ENABLE
  return status;
}

static iree_status_t loom_low_emission_frame_lower_spill_traffic(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* options,
    loom_module_t* module, loom_op_t* low_func_op,
    loom_low_emission_frame_lower_spill_traffic_result_t* out_result,
    iree_arena_allocator_t* arena, loom_low_planning_statistics_t* statistics) {
  *out_result = (loom_low_emission_frame_lower_spill_traffic_result_t){0};
  if (options->lower_spill_traffic == NULL) {
    return iree_ok_status();
  }
  if (statistics != NULL) ++statistics->repair.spill_traffic_lowering_count;
  return options->lower_spill_traffic(
      options->lower_spill_traffic_user_data, module, low_func_op,
      frame_options->emitter, arena, out_result);
}

static iree_status_t loom_low_emission_frame_append_materialized_spill_records(
    const loom_low_allocation_materialized_spill_t* records,
    iree_host_size_t record_count,
    loom_low_allocation_materialized_spill_list_t* list,
    iree_arena_allocator_t* arena) {
  if (record_count == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_materialized_spill_vec_t* vec = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*vec), (void**)&vec));
  loom_low_allocation_materialized_spill_t* record_copy = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, record_count, sizeof(*record_copy), (void**)&record_copy));
  memcpy(record_copy, records, record_count * sizeof(*record_copy));
  *vec = (loom_low_allocation_materialized_spill_vec_t){
      .records = record_copy,
      .record_count = record_count,
  };
  if (list->tail) {
    list->tail->next = vec;
  } else {
    list->head = vec;
  }
  list->tail = vec;
  if (!iree_host_size_checked_add(list->record_count, record_count,
                                  &list->record_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low emission frame materialized spill record count overflows host "
        "size");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_copy_pair_uses(
    loom_low_placement_pair_use_list_t source, iree_arena_allocator_t* arena,
    loom_low_placement_pair_use_list_t* out_copy) {
  *out_copy = source;
  if (source.count == 0) return iree_ok_status();
  loom_low_placement_pair_use_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source.count, sizeof(*values), (void**)&values));
  memcpy(values, source.values, source.count * sizeof(*values));
  out_copy->values = values;
  return iree_ok_status();
}

static void loom_low_emission_frame_accumulate_materialization(
    const loom_low_allocation_materialization_result_t* result,
    loom_low_emission_frame_materialization_summary_t* summary) {
  summary->spill_storage_count += result->storage_count;
  summary->spill_storage_bytes += result->storage_bytes;
  summary->spill_store_count += result->spill_count;
  summary->spill_store_bytes += result->spill_bytes;
  summary->reload_count += result->reload_count;
  summary->reload_bytes += result->reload_bytes;
}

static iree_status_t loom_low_emission_frame_validate_final(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* options,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    bool* out_accepted) {
  *out_accepted = false;
  loom_low_addressability_validation_result_t addressability_result = {0};
  IREE_RETURN_IF_ERROR(loom_low_addressability_validate_allocated_packets(
      &frame->schedule, &frame->allocation, frame_options->emitter,
      &addressability_result));
  if (addressability_result.error_count != 0) {
    return iree_ok_status();
  }
  if (options->validate_frame == NULL) {
    *out_accepted = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      options->validate_frame(options->validate_frame_user_data, frame, arena));
  *out_accepted = true;
  return iree_ok_status();
}

static uint64_t loom_low_emission_frame_residency_resource_units(
    const loom_target_residency_derived_resource_table_t* resource_table,
    uint16_t resource_id, const uint32_t* units_by_reg_class,
    iree_host_size_t reg_class_count) {
  const loom_target_residency_derived_resource_t* resource =
      &resource_table->resources[resource_id];
  uint64_t resource_units = 0;
  for (uint16_t i = 0; i < resource->member_count; ++i) {
    const loom_target_residency_derived_member_t* member =
        &resource_table->members[resource->member_start + i];
    IREE_ASSERT_EQ(member->resource_id, resource_id);
    IREE_ASSERT_LT(member->direct_resource_id, reg_class_count);
    const uint64_t contribution = loom_target_residency_round_resource_units(
        units_by_reg_class[member->direct_resource_id],
        member->contribution_granularity);
    if (resource_units > UINT64_MAX - contribution) return UINT64_MAX;
    resource_units += contribution;
  }
  return resource_units;
}

static bool loom_low_emission_frame_crosses_new_pressure_cliff(
    const loom_target_residency_model_t* residency_model,
    const uint32_t* baseline_units_by_reg_class,
    const loom_low_allocation_table_t* allocation) {
  if (loom_target_residency_model_is_empty(residency_model)) return false;
  const loom_target_residency_direct_resource_table_t* direct_resources =
      &residency_model->direct_resources;
  IREE_ASSERT_EQ(direct_resources->resource_count,
                 allocation->physical_extents.count);
  for (uint16_t resource_id = 0; resource_id < direct_resources->resource_count;
       ++resource_id) {
    const loom_target_residency_cliff_range_t range =
        loom_target_residency_direct_resource_cliff_range(direct_resources,
                                                          resource_id);
    const loom_target_residency_cliff_t* cliffs =
        range.count == 0 ? NULL : &direct_resources->cliffs[range.start];
    loom_target_residency_cliff_evaluation_t baseline_evaluation;
    loom_target_residency_evaluate_cliffs(
        cliffs, range.count, residency_model->best_tier,
        baseline_units_by_reg_class[resource_id], &baseline_evaluation);
    loom_target_residency_cliff_evaluation_t allocated_evaluation;
    loom_target_residency_evaluate_cliffs(
        cliffs, range.count, residency_model->best_tier,
        allocation->physical_extents.ends_by_reg_class[resource_id],
        &allocated_evaluation);
    if (allocated_evaluation.tier < baseline_evaluation.tier) return true;
  }
  const loom_target_residency_derived_resource_table_t* resource_table =
      &residency_model->derived_resources;
  for (uint16_t resource_id = 0; resource_id < resource_table->resource_count;
       ++resource_id) {
    const loom_target_residency_derived_resource_t* resource =
        &resource_table->resources[resource_id];
    const uint64_t baseline_resource_units =
        loom_low_emission_frame_residency_resource_units(
            resource_table, resource_id, baseline_units_by_reg_class,
            allocation->physical_extents.count);
    const uint64_t allocated_resource_units =
        loom_low_emission_frame_residency_resource_units(
            resource_table, resource_id,
            allocation->physical_extents.ends_by_reg_class,
            allocation->physical_extents.count);
    const loom_target_residency_cliff_t* cliffs =
        resource->cliff_count == 0
            ? NULL
            : &resource_table->cliffs[resource->cliff_start];
    loom_target_residency_cliff_evaluation_t baseline_evaluation;
    loom_target_residency_evaluate_cliffs(
        cliffs, resource->cliff_count, residency_model->best_tier,
        baseline_resource_units, &baseline_evaluation);
    loom_target_residency_cliff_evaluation_t allocated_evaluation;
    loom_target_residency_evaluate_cliffs(
        cliffs, resource->cliff_count, residency_model->best_tier,
        allocated_resource_units, &allocated_evaluation);
    if (allocated_evaluation.tier < baseline_evaluation.tier) return true;
  }
  return false;
}

static iree_status_t loom_low_emission_frame_emit_rematerialization_decision(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_allocation_table_t* allocation,
    loom_low_allocation_rematerialization_trigger_t trigger,
    const loom_low_allocation_rematerialization_result_t* result) {
  if (!iree_any_bit_set(frame_options->allocation_diagnostic_flags,
                        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS)) {
    return iree_ok_status();
  }
  return loom_low_allocation_rematerialization_emit_decision(
      allocation, trigger, result, frame_options->emitter);
}

static iree_status_t loom_low_emission_frame_emit_live_range_split_decision(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_allocation_table_t* allocation,
    loom_low_allocation_live_range_split_trigger_t trigger,
    const loom_low_allocation_live_range_split_result_t* result) {
  if (!iree_any_bit_set(frame_options->allocation_diagnostic_flags,
                        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS)) {
    return iree_ok_status();
  }
  return loom_low_allocation_live_range_split_emit_decision(
      allocation, trigger, result, frame_options->emitter);
}

static iree_string_view_t loom_low_emission_frame_failure_code(
    loom_low_emission_frame_failure_t failure) {
  switch (failure) {
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS:
      return IREE_SV("remaining-spill-assignments");
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT:
      return IREE_SV("spill-materialization-iteration-limit");
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS:
      return IREE_SV("spill-materialization-no-progress");
    default:
      return IREE_SV("<unknown>");
  }
}

static iree_status_t loom_low_emission_frame_fail_final(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_t* frame,
    loom_low_emission_frame_failure_t failure, iree_host_size_t iteration_count,
    iree_host_size_t iteration_limit, loom_low_emission_frame_t* out_frame) {
  *out_frame = *frame;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&frame->target)),
      loom_param_string(loom_low_diagnostic_export_name(&frame->target)),
      loom_param_string(loom_low_diagnostic_config_key(&frame->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(frame->module, frame->function_op)),
      loom_param_string(loom_low_emission_frame_failure_code(failure)),
      loom_param_u64((uint64_t)iteration_count),
      loom_param_u64((uint64_t)iteration_limit),
      loom_param_u64((uint64_t)frame->allocation.spill_plan_count),
      loom_param_u64((uint64_t)frame->allocation.spill_count),
      loom_param_u64((uint64_t)frame->schedule.scheduled_node_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = frame->function_op,
      .error = LOOM_ERR_BACKEND_021,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(frame_options->emitter, &emission);
}

static iree_status_t loom_low_emission_frame_apply_materialization_summary(
    const loom_low_emission_frame_materialization_summary_t* summary,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* frame) {
  frame->materialized_spill_storage_count = summary->spill_storage_count;
  frame->materialized_spill_storage_bytes = summary->spill_storage_bytes;
  frame->materialized_spill_store_count = summary->spill_store_count;
  frame->materialized_spill_store_bytes = summary->spill_store_bytes;
  frame->materialized_reload_count = summary->reload_count;
  frame->materialized_reload_bytes = summary->reload_bytes;
  for (const loom_low_allocation_materialized_spill_vec_t* vec =
           summary->spill_records.head;
       vec != NULL; vec = vec->next) {
    IREE_RETURN_IF_ERROR(
        loom_low_emission_frame_append_materialized_spill_records(
            vec->records, vec->record_count, &frame->materialized_spills,
            arena));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_replay_diagnostics(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* frame_options,
    loom_low_placement_pair_use_list_t preferred_pair_uses,
    iree_bitmap_t required_register_values,
    const iree_arena_checkpoint_t* frame_checkpoint,
    iree_arena_allocator_t* repair_arena, iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* arena, loom_low_planning_statistics_t* statistics,
    loom_low_emission_frame_t* out_frame) {
  loom_low_emission_frame_record_memory_high_water(
      frame_checkpoint, repair_arena, scratch_arena, statistics);
  iree_arena_checkpoint_restore(frame_checkpoint);
  *out_frame = (loom_low_emission_frame_t){0};
  if (statistics != NULL) {
    ++statistics->repair.diagnostic_replay_count;
  }
  return loom_low_emission_frame_build_with_diagnostic_emitter(
      module, low_func_op, frame_options, preferred_pair_uses,
      required_register_values, frame_options->emitter, arena, statistics,
      out_frame);
}

static bool loom_low_emission_frame_diagnostics_requested(
    const loom_low_emission_frame_options_t* frame_options) {
  return frame_options->emitter.fn != NULL &&
         (frame_options->schedule_diagnostic_flags != 0 ||
          frame_options->allocation_diagnostic_flags != 0);
}

static iree_status_t loom_low_emission_frame_build_spill_free_impl(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    const iree_arena_checkpoint_t* frame_checkpoint,
    iree_arena_allocator_t* repair_arena, iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* arena, loom_low_planning_statistics_t* statistics,
    loom_low_emission_frame_t* out_frame) {
  *out_frame = (loom_low_emission_frame_t){0};
  if (spill_free_options == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "spill-free low emission frame construction requires spill-free "
        "options");
  }
  const loom_target_residency_model_t* residency_model =
      frame_options->residency_model;

  // Target-lowered spill helpers require registers. Rematerialized clones
  // inherit that requirement so rematerialization cannot erase the fact.
  iree_bitmap_t required_register_values = {0};
  loom_low_emission_frame_lower_spill_traffic_result_t spill_lowering_result = {
      0};
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_lower_spill_traffic(
      frame_options, spill_free_options, module, low_func_op,
      &spill_lowering_result, scratch_arena, statistics));
  if (spill_lowering_result.error_count != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_record_value_list(
      spill_lowering_result.required_register_value_ids,
      spill_lowering_result.required_register_value_count, repair_arena,
      &required_register_values));
  iree_host_size_t repair_iteration_count = 0;
  iree_host_size_t value_repair_iteration_count = 0;
  iree_host_size_t spill_materialization_iteration_count = 0;
  iree_host_size_t last_repaired_spill_plan_count = IREE_HOST_SIZE_MAX;
  bool restore_frame_before_build = false;
  bool pair_replication_attempted = false;
  loom_low_placement_pair_use_list_t pair_replication_preferred_pairs =
      loom_low_placement_pair_use_list_empty();
  iree_host_size_t pair_replication_baseline_packet_move_count = 0;
  uint32_t* pair_replication_baseline_units_by_reg_class = NULL;
  loom_low_allocation_pair_replication_result_t pair_replication = {0};
  loom_low_emission_frame_materialization_summary_t materialization_summary = {
      0};
  for (;;) {
    loom_low_emission_frame_record_memory_high_water(
        frame_checkpoint, repair_arena, scratch_arena, statistics);
    if (restore_frame_before_build) {
      iree_arena_checkpoint_restore(frame_checkpoint);
      restore_frame_before_build = false;
    }
    iree_arena_reset(scratch_arena);
    loom_low_emission_frame_t frame = {0};
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_with_diagnostic_emitter(
        module, low_func_op, frame_options,
        pair_replication.edit_count != 0
            ? pair_replication_preferred_pairs
            : loom_low_placement_pair_use_list_empty(),
        required_register_values, (iree_diagnostic_emitter_t){0}, arena,
        statistics, &frame));
    if (pair_replication.edit_count != 0) {
      bool rejected =
          frame.schedule.error_count != 0 ||
          frame.allocation.error_count != 0 ||
          frame.allocation.spill_plan_count != 0 ||
          frame.allocation.spill_count != 0 ||
          loom_low_emission_frame_crosses_new_pressure_cliff(
              residency_model, pair_replication_baseline_units_by_reg_class,
              &frame.allocation);
      if (!rejected) {
        uint64_t satisfied_packet_savings = 0;
        IREE_RETURN_IF_ERROR(loom_low_allocation_satisfied_pair_packet_savings(
            &frame.allocation, frame.schedule.placement_pair_uses,
            &satisfied_packet_savings));
        // Cross-add to compare net savings without signed arithmetic:
        // new_savings - new_moves > baseline_savings - baseline_moves.
        const uint64_t baseline_net_side =
            pair_replication.baseline_satisfied_packet_savings +
            frame.allocation.packet_move_count;
        const uint64_t replicated_net_side =
            satisfied_packet_savings +
            pair_replication_baseline_packet_move_count;
        rejected = replicated_net_side <= baseline_net_side;
      }
      if (rejected) {
        if (statistics != NULL) {
          ++statistics->repair.pair_replication_rejection_count;
        }
        IREE_RETURN_IF_ERROR(loom_low_allocation_rollback_pair_replication(
            module, &pair_replication, scratch_arena));
        pair_replication = (loom_low_allocation_pair_replication_result_t){0};
        loom_low_emission_frame_advance_repair_iteration(
            &repair_iteration_count, statistics);
        restore_frame_before_build = true;
        continue;
      }
    }
    if (frame.schedule.error_count != 0) {
      if (value_repair_iteration_count <
              LOOM_LOW_EMISSION_FRAME_MAX_VALUE_REPAIR_ITERATIONS &&
          frame.schedule.failure.state_value_id != LOOM_VALUE_ID_INVALID) {
        loom_low_value_rematerialization_result_t result = {0};
        const iree_host_size_t first_rematerialized_value_id =
            module->values.count;
        IREE_RETURN_IF_ERROR(loom_low_rematerialize_value_uses(
            module, &frame.schedule.target,
            frame.schedule.failure.state_value_id, scratch_arena, &result));
        if (result.rewritten_operand_count != 0) {
          IREE_RETURN_IF_ERROR(
              loom_low_emission_frame_record_derived_register_requirement(
                  frame.schedule.failure.state_value_id,
                  first_rematerialized_value_id, module->values.count,
                  repair_arena, &required_register_values));
          if (statistics != NULL) {
            statistics->repair.rematerialized_operand_count +=
                result.rewritten_operand_count;
          }
          ++value_repair_iteration_count;
          loom_low_emission_frame_advance_repair_iteration(
              &repair_iteration_count, statistics);
          restore_frame_before_build = true;
          continue;
        }
      }
      if (frame_options->emitter.fn != NULL) {
        IREE_RETURN_IF_ERROR(loom_low_emission_frame_replay_diagnostics(
            module, low_func_op, frame_options,
            loom_low_placement_pair_use_list_empty(), required_register_values,
            frame_checkpoint, repair_arena, scratch_arena, arena, statistics,
            &frame));
      }
      *out_frame = frame;
      return iree_ok_status();
    }
    if (frame.allocation.error_count != 0) {
      if (value_repair_iteration_count <
          LOOM_LOW_EMISSION_FRAME_MAX_VALUE_REPAIR_ITERATIONS) {
        loom_low_allocation_rematerialization_result_t result = {0};
        const iree_host_size_t first_rematerialized_value_id =
            module->values.count;
        IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_failure(
            module, &frame.allocation, scratch_arena, &result));
        if (result.value.rewritten_operand_count != 0) {
          IREE_RETURN_IF_ERROR(
              loom_low_emission_frame_record_derived_register_requirement(
                  result.value.value_id, first_rematerialized_value_id,
                  module->values.count, repair_arena,
                  &required_register_values));
          if (statistics != NULL) {
            statistics->repair.rematerialized_operand_count +=
                result.value.rewritten_operand_count;
          }
          IREE_RETURN_IF_ERROR(
              loom_low_emission_frame_emit_rematerialization_decision(
                  frame_options, &frame.allocation,
                  LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE,
                  &result));
          ++value_repair_iteration_count;
          loom_low_emission_frame_advance_repair_iteration(
              &repair_iteration_count, statistics);
          restore_frame_before_build = true;
          continue;
        }
      }
      if (frame_options->emitter.fn != NULL) {
        IREE_RETURN_IF_ERROR(loom_low_emission_frame_replay_diagnostics(
            module, low_func_op, frame_options,
            loom_low_placement_pair_use_list_empty(), required_register_values,
            frame_checkpoint, repair_arena, scratch_arena, arena, statistics,
            &frame));
      }
      *out_frame = frame;
      return iree_ok_status();
    }

    if (frame.allocation.spill_plan_count == 0 &&
        frame.allocation.spill_count == 0) {
      if (!pair_replication_attempted) {
        pair_replication_attempted = true;
        if (statistics != NULL) {
          ++statistics->repair.pair_replication_attempt_count;
        }
        IREE_RETURN_IF_ERROR(loom_low_allocation_replicate_pair_sources(
            module, &frame.allocation, frame.schedule.placement_pair_uses,
            repair_arena, &pair_replication));
        if (statistics != NULL) {
          statistics->repair.pair_replication_edit_count +=
              pair_replication.edit_count;
        }
        if (pair_replication.edit_count != 0) {
          IREE_RETURN_IF_ERROR(loom_low_emission_frame_copy_pair_uses(
              frame.schedule.placement_pair_uses, repair_arena,
              &pair_replication_preferred_pairs));
          if (!loom_target_residency_model_is_empty(residency_model)) {
            IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
                repair_arena, frame.allocation.physical_extents.count,
                sizeof(*pair_replication_baseline_units_by_reg_class),
                (void**)&pair_replication_baseline_units_by_reg_class));
            memcpy(pair_replication_baseline_units_by_reg_class,
                   frame.allocation.physical_extents.ends_by_reg_class,
                   frame.allocation.physical_extents.count *
                       sizeof(*pair_replication_baseline_units_by_reg_class));
          }
          pair_replication_baseline_packet_move_count =
              frame.allocation.packet_move_count;
          loom_low_emission_frame_advance_repair_iteration(
              &repair_iteration_count, statistics);
          restore_frame_before_build = true;
          continue;
        }
      }
      bool accepted = false;
      IREE_RETURN_IF_ERROR(loom_low_emission_frame_validate_final(
          frame_options, spill_free_options, &frame, scratch_arena, &accepted));
      if (!accepted) {
        return iree_ok_status();
      }
      if (loom_low_emission_frame_diagnostics_requested(frame_options)) {
        IREE_RETURN_IF_ERROR(loom_low_emission_frame_replay_diagnostics(
            module, low_func_op, frame_options,
            pair_replication.edit_count != 0
                ? pair_replication_preferred_pairs
                : loom_low_placement_pair_use_list_empty(),
            required_register_values, frame_checkpoint, repair_arena,
            scratch_arena, arena, statistics, &frame));
      }
      IREE_RETURN_IF_ERROR(
          loom_low_emission_frame_apply_materialization_summary(
              &materialization_summary, arena, &frame));
      *out_frame = frame;
      return iree_ok_status();
    }
    if (frame.allocation.spill_plan_count == 0) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS,
          spill_materialization_iteration_count,
          LOOM_LOW_EMISSION_FRAME_MAX_SPILL_MATERIALIZATION_ITERATIONS,
          out_frame);
    }
    if (value_repair_iteration_count <
            LOOM_LOW_EMISSION_FRAME_MAX_VALUE_REPAIR_ITERATIONS &&
        frame.allocation.spill_plan_count < last_repaired_spill_plan_count) {
      loom_low_allocation_rematerialization_result_t rematerialization_result =
          {0};
      const iree_host_size_t first_rematerialized_value_id =
          module->values.count;
      IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_spill_plan(
          module, &frame.allocation, scratch_arena, &rematerialization_result));
      if (rematerialization_result.value.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_emission_frame_record_derived_register_requirement(
                rematerialization_result.value.value_id,
                first_rematerialized_value_id, module->values.count,
                repair_arena, &required_register_values));
        if (statistics != NULL) {
          statistics->repair.rematerialized_operand_count +=
              rematerialization_result.value.rewritten_operand_count;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_emission_frame_emit_rematerialization_decision(
                frame_options, &frame.allocation,
                LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_SPILL_PLAN,
                &rematerialization_result));
        last_repaired_spill_plan_count = frame.allocation.spill_plan_count;
        ++value_repair_iteration_count;
        loom_low_emission_frame_advance_repair_iteration(
            &repair_iteration_count, statistics);
        restore_frame_before_build = true;
        continue;
      }
      loom_low_allocation_live_range_split_result_t split_result = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_split_fixed_value_spill_plan(
          module, &frame.allocation, scratch_arena, &split_result));
      if (statistics != NULL) {
        statistics->repair.live_range_split_operand_count +=
            split_result.rewritten_operand_count;
      }
      if (split_result.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_emission_frame_emit_live_range_split_decision(
                frame_options, &frame.allocation,
                LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN,
                &split_result));
        last_repaired_spill_plan_count = frame.allocation.spill_plan_count;
        ++value_repair_iteration_count;
        loom_low_emission_frame_advance_repair_iteration(
            &repair_iteration_count, statistics);
        restore_frame_before_build = true;
        continue;
      }
    }
    if (spill_materialization_iteration_count >=
        LOOM_LOW_EMISSION_FRAME_MAX_SPILL_MATERIALIZATION_ITERATIONS) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT,
          spill_materialization_iteration_count,
          LOOM_LOW_EMISSION_FRAME_MAX_SPILL_MATERIALIZATION_ITERATIONS,
          out_frame);
    }

    loom_low_allocation_materialization_result_t result = {0};
    loom_low_allocation_materialization_options_t materialization_options =
        spill_free_options->materialization_options;
    // The frame loop materializes one complete allocation snapshot at a time;
    // the next iteration accounts for any spill traffic introduced by it.
    materialization_options.max_spill_plan_count = 0;
    if (materialization_options.emitter.fn == NULL) {
      materialization_options.emitter = frame_options->emitter;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_spills(
        &frame.allocation, &materialization_options, scratch_arena, &result));
    if (result.error_count != 0) {
      return iree_ok_status();
    }
    if (result.storage_count == 0 && result.spill_count == 0 &&
        result.reload_count == 0) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS,
          spill_materialization_iteration_count,
          LOOM_LOW_EMISSION_FRAME_MAX_SPILL_MATERIALIZATION_ITERATIONS,
          out_frame);
    }
    if (statistics != NULL) {
      ++statistics->repair.spill_materialization_batch_count;
    }
    loom_low_emission_frame_accumulate_materialization(
        &result, &materialization_summary);
    IREE_RETURN_IF_ERROR(
        loom_low_emission_frame_append_materialized_spill_records(
            result.materialized_spills, result.materialized_spill_count,
            &materialization_summary.spill_records, repair_arena));
    last_repaired_spill_plan_count = IREE_HOST_SIZE_MAX;

    loom_low_emission_frame_record_memory_high_water(
        frame_checkpoint, repair_arena, scratch_arena, statistics);
    iree_arena_reset(scratch_arena);
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_lower_spill_traffic(
        frame_options, spill_free_options, module, low_func_op,
        &spill_lowering_result, scratch_arena, statistics));
    if (spill_lowering_result.error_count != 0) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_record_value_list(
        spill_lowering_result.required_register_value_ids,
        spill_lowering_result.required_register_value_count, repair_arena,
        &required_register_values));
    ++spill_materialization_iteration_count;
    loom_low_emission_frame_advance_repair_iteration(&repair_iteration_count,
                                                     statistics);
    restore_frame_before_build = true;
  }
}

iree_status_t loom_low_emission_frame_build_spill_free(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  IREE_ASSERT_ARGUMENT(frame_options);
  IREE_ASSERT(frame_options->emitter.fn != NULL);
  loom_low_planning_statistics_t* statistics = frame_options->statistics;
  if (statistics != NULL) {
    *statistics = (loom_low_planning_statistics_t){0};
  }
#if IREE_STATISTICS_ENABLE
  iree_arena_block_pool_statistics_t pool_statistics_before = {0};
  if (statistics != NULL) {
    iree_arena_block_pool_query_statistics(arena->block_pool,
                                           &pool_statistics_before);
  }
#endif  // IREE_STATISTICS_ENABLE
  const iree_arena_checkpoint_t frame_checkpoint =
      iree_arena_checkpoint_save(arena);
  iree_arena_allocator_t repair_arena;
  iree_arena_initialize(arena->block_pool, &repair_arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status = loom_low_emission_frame_build_spill_free_impl(
      module, low_func_op, frame_options, spill_free_options, &frame_checkpoint,
      &repair_arena, &scratch_arena, arena, statistics, out_frame);
  loom_low_emission_frame_record_memory_high_water(
      &frame_checkpoint, &repair_arena, &scratch_arena, statistics);
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&repair_arena);
  if (statistics != NULL) {
#if IREE_STATISTICS_ENABLE
    loom_low_emission_frame_record_system_allocation_delta(
        arena->block_pool, &pool_statistics_before, statistics);
#endif  // IREE_STATISTICS_ENABLE
  }
  return status;
}
