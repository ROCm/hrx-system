// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame.h"

#include <string.h>

#include "loom/codegen/low/addressability.h"
#include "loom/codegen/low/allocation_live_range_splitting.h"
#include "loom/codegen/low/allocation_rematerialization.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/low/ops.h"

typedef enum loom_low_emission_frame_failure_e {
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS = 0,
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT = 1,
  LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT = 2,
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS = 3,
} loom_low_emission_frame_failure_t;

// Emission-frame repairs rebuild whole-function scheduling and allocation.
// Every repair callback operates on a complete snapshot, so reaching this
// limit indicates a non-convergent strategy rather than useful fine-grained
// progress.
#define LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS 8

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

static iree_status_t loom_low_emission_frame_liveness_order_from_schedule(
    loom_op_t* low_func_op, const loom_low_schedule_table_t* schedule,
    iree_arena_allocator_t* arena, loom_liveness_order_t* out_order) {
  *out_order = loom_liveness_order_empty();

  loom_region_t* body = loom_low_function_body(low_func_op);
  if (schedule->function_op != low_func_op ||
      schedule->block_count != body->block_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low emission-frame schedule must describe the low function body");
  }
  if (schedule->scheduled_node_count != 0 &&
      schedule->scheduled_node_indices == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low emission-frame schedule has packets but no packet index table");
  }

  loom_liveness_block_order_t* block_orders = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, body->block_count, sizeof(*block_orders), (void**)&block_orders));
  iree_host_size_t total_scheduled_nodes = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_low_schedule_block_t* schedule_block =
        &schedule->blocks[block_index];
    if (schedule_block->block != block ||
        schedule_block->scheduled_node_count != block->op_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low emission-frame schedule block %u does not "
                              "match the function body",
                              block_index);
    }
    const loom_op_t** ops = NULL;
    if (schedule_block->scheduled_node_count != 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, schedule_block->scheduled_node_count,
                                    sizeof(*ops), (void**)&ops));
    }
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < schedule_block->scheduled_node_count;
         ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)schedule_block->scheduled_node_start +
          scheduled_ordinal;
      if (packet_index >= schedule->scheduled_node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low emission-frame schedule block %u references packet %" PRIhsz
            " outside the schedule",
            block_index, packet_index);
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low emission-frame schedule packet %" PRIhsz
                                " references node %" PRIu32,
                                packet_index, node_index);
      }
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      if (node->block != block ||
          node->scheduled_ordinal != scheduled_ordinal) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low emission-frame schedule node does not match its block order");
      }
      ops[scheduled_ordinal] = node->op;
      ++total_scheduled_nodes;
    }
    block_orders[block_index] = (loom_liveness_block_order_t){
        .block = block,
        .ops = ops,
        .op_count = schedule_block->scheduled_node_count,
    };
  }
  if (total_scheduled_nodes != schedule->scheduled_node_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low emission-frame schedule has %" PRIhsz
        " scheduled packet(s) but body blocks cover %" PRIhsz,
        schedule->scheduled_node_count, total_scheduled_nodes);
  }
  *out_order = (loom_liveness_order_t){
      .blocks = block_orders,
      .block_count = body->block_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_build_with_diagnostic_emitter(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    loom_low_placement_pair_use_list_t preferred_pair_uses,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_low_emission_frame_t* out_frame) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }

  *out_frame = (loom_low_emission_frame_t){
      .module = module,
      .function_op = low_func_op,
  };

  loom_low_schedule_options_t schedule_options = {
      .descriptor_registry = options->descriptor_registry,
      .target_selection = options->target_selection,
      .memory_access_table = options->memory_access_table,
      .pressure_cliffs = options->pressure_cliffs,
      .allocation_budgets = options->allocation_budgets,
      .allocation_budget_count = options->allocation_budget_count,
      .pair_affinities = options->schedule_pair_affinities,
      .preferred_pair_uses = preferred_pair_uses,
      .structural_state_reads = options->schedule_structural_state_reads,
      .emitter = diagnostic_emitter,
      .diagnostic_flags = options->schedule_diagnostic_flags,
      .strategy = options->schedule_strategy,
  };
  IREE_RETURN_IF_ERROR(loom_low_schedule_function(
      module, low_func_op, &schedule_options, arena, &out_frame->schedule));
  if (out_frame->schedule.error_count != 0) {
    out_frame->target = out_frame->schedule.target;
    return iree_ok_status();
  }

  loom_low_storage_lease_table_t storage_leases = {0};
  if (options->storage_lease_provider != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_storage_lease_build(
        &out_frame->schedule, options->storage_lease_provider, arena,
        &storage_leases));
  }

  loom_liveness_order_t liveness_order = loom_liveness_order_empty();
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_liveness_order_from_schedule(
      low_func_op, &out_frame->schedule, arena, &liveness_order));
  loom_low_allocation_options_t allocation_options = {
      .liveness_order = liveness_order,
      .descriptor_registry = options->descriptor_registry,
      .target_selection = options->target_selection,
      .budgets = options->allocation_budgets,
      .budget_count = options->allocation_budget_count,
      .fixed_values = options->allocation_fixed_values,
      .fixed_value_count = options->allocation_fixed_value_count,
      .reserved_ranges = options->allocation_reserved_ranges,
      .reserved_range_count = options->allocation_reserved_range_count,
      .storage_leases = storage_leases,
      .placement_pair_uses = out_frame->schedule.placement_pair_uses,
      .emitter = diagnostic_emitter,
      .diagnostic_flags = options->allocation_diagnostic_flags,
  };
  IREE_RETURN_IF_ERROR(loom_low_allocate_function(
      module, low_func_op, &allocation_options, arena, &out_frame->allocation));

  if (out_frame->allocation.error_count != 0) {
    out_frame->target = out_frame->schedule.target;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(&out_frame->schedule,
                                                       &out_frame->allocation));
  out_frame->target = out_frame->schedule.target;
  return iree_ok_status();
}

iree_status_t loom_low_emission_frame_build(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  return loom_low_emission_frame_build_with_diagnostic_emitter(
      module, low_func_op, options, loom_low_placement_pair_use_list_empty(),
      options->emitter, arena, out_frame);
}

static iree_status_t loom_low_emission_frame_lower_spill_traffic(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* options,
    loom_module_t* module, loom_op_t* low_func_op,
    loom_low_emission_frame_lower_spill_traffic_result_t* out_result,
    iree_arena_allocator_t* arena) {
  *out_result = (loom_low_emission_frame_lower_spill_traffic_result_t){0};
  if (options->lower_spill_traffic == NULL) {
    return iree_ok_status();
  }
  return options->lower_spill_traffic(
      options->lower_spill_traffic_user_data, module, low_func_op,
      frame_options->emitter, arena, out_result);
}

static iree_status_t loom_low_emission_frame_materialize_address_state(
    const loom_low_emission_frame_spill_free_options_t* options,
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result) {
  *out_result = (loom_low_emission_frame_materialize_address_state_result_t){0};
  if (options->materialize_address_state == NULL) {
    return iree_ok_status();
  }
  return options->materialize_address_state(
      options->materialize_address_state_user_data, module, low_func_op, frame,
      arena, out_result);
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

static uint32_t loom_low_emission_frame_allocated_units_for_reg_class(
    const loom_low_allocation_table_t* allocation, uint16_t reg_class_id) {
  uint32_t allocated_units = 0;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->descriptor_reg_class_id != reg_class_id ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    const uint64_t location_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    IREE_ASSERT_LE(location_end, UINT32_MAX);
    allocated_units = iree_max(allocated_units, (uint32_t)location_end);
  }
  return allocated_units;
}

static void loom_low_emission_frame_capture_pressure_cliff_units(
    loom_low_pressure_cliff_table_t pressure_cliffs,
    const loom_low_allocation_table_t* allocation, uint32_t* out_units) {
  for (iree_host_size_t i = 0; i < pressure_cliffs.count; ++i) {
    out_units[i] = loom_low_emission_frame_allocated_units_for_reg_class(
        allocation, pressure_cliffs.values[i].descriptor_reg_class_id);
  }
}

static bool loom_low_emission_frame_crosses_new_pressure_cliff(
    loom_low_pressure_cliff_table_t pressure_cliffs,
    const uint32_t* baseline_units,
    const loom_low_allocation_table_t* allocation) {
  for (iree_host_size_t i = 0; i < pressure_cliffs.count; ++i) {
    const loom_low_pressure_cliff_t* cliff = &pressure_cliffs.values[i];
    if (baseline_units[i] >= cliff->cliff_units) {
      continue;
    }
    const uint32_t allocated_units =
        loom_low_emission_frame_allocated_units_for_reg_class(
            allocation, cliff->descriptor_reg_class_id);
    if (allocated_units >= cliff->cliff_units) {
      return true;
    }
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
    case LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT:
      return IREE_SV("address-state-iteration-limit");
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
    iree_host_size_t iteration_limit) {
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

static iree_status_t loom_low_emission_frame_build_spill_free_impl(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    const iree_arena_checkpoint_t* frame_checkpoint,
    iree_arena_allocator_t* repair_arena, iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  *out_frame = (loom_low_emission_frame_t){0};
  if (spill_free_options == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "spill-free low emission frame construction requires spill-free "
        "options");
  }

  loom_low_emission_frame_lower_spill_traffic_result_t spill_lowering_result = {
      0};
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_lower_spill_traffic(
      frame_options, spill_free_options, module, low_func_op,
      &spill_lowering_result, scratch_arena));
  if (spill_lowering_result.error_count != 0) {
    return iree_ok_status();
  }
  iree_host_size_t repair_iteration_count = 0;
  iree_host_size_t last_repaired_spill_plan_count = IREE_HOST_SIZE_MAX;
  bool restore_frame_before_build = false;
  bool pair_replication_attempted = false;
  loom_low_placement_pair_use_list_t pair_replication_preferred_pairs =
      loom_low_placement_pair_use_list_empty();
  iree_host_size_t pair_replication_baseline_packet_move_count = 0;
  uint32_t* pair_replication_baseline_cliff_units = NULL;
  loom_low_allocation_pair_replication_result_t pair_replication = {0};
  loom_low_emission_frame_materialization_summary_t materialization_summary = {
      0};
  for (;;) {
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
        (iree_diagnostic_emitter_t){0}, arena, &frame));
    if (pair_replication.edit_count != 0) {
      bool rejected =
          frame.schedule.error_count != 0 ||
          frame.allocation.error_count != 0 ||
          frame.allocation.spill_plan_count != 0 ||
          frame.allocation.spill_count != 0 ||
          loom_low_emission_frame_crosses_new_pressure_cliff(
              frame_options->pressure_cliffs,
              pair_replication_baseline_cliff_units, &frame.allocation);
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
        IREE_RETURN_IF_ERROR(loom_low_allocation_rollback_pair_replication(
            module, &pair_replication, scratch_arena));
        pair_replication = (loom_low_allocation_pair_replication_result_t){0};
        ++repair_iteration_count;
        restore_frame_before_build = true;
        continue;
      }
    }
    if (frame.schedule.error_count != 0) {
      if (frame_options->emitter.fn != NULL) {
        iree_arena_checkpoint_restore(frame_checkpoint);
        frame = (loom_low_emission_frame_t){0};
        IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
            module, low_func_op, frame_options, arena, &frame));
      }
      *out_frame = frame;
      return iree_ok_status();
    }
    if (frame.allocation.error_count != 0) {
      if (repair_iteration_count <
          LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS) {
        loom_low_allocation_rematerialization_result_t result = {0};
        IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_failure(
            module, &frame.allocation, scratch_arena, &result));
        if (result.rewritten_operand_count != 0) {
          IREE_RETURN_IF_ERROR(
              loom_low_emission_frame_emit_rematerialization_decision(
                  frame_options, &frame.allocation,
                  LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE,
                  &result));
          ++repair_iteration_count;
          restore_frame_before_build = true;
          continue;
        }
      }
      if (frame_options->emitter.fn != NULL) {
        iree_arena_checkpoint_restore(frame_checkpoint);
        frame = (loom_low_emission_frame_t){0};
        IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
            module, low_func_op, frame_options, arena, &frame));
      }
      *out_frame = frame;
      return iree_ok_status();
    }

    if (frame.allocation.spill_plan_count == 0 &&
        frame.allocation.spill_count == 0) {
      if (!pair_replication_attempted) {
        pair_replication_attempted = true;
        if (repair_iteration_count + 2 <=
            LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS) {
          IREE_RETURN_IF_ERROR(loom_low_allocation_replicate_pair_sources(
              module, &frame.allocation, frame.schedule.placement_pair_uses,
              repair_arena, &pair_replication));
        }
        if (pair_replication.edit_count != 0) {
          IREE_RETURN_IF_ERROR(loom_low_emission_frame_copy_pair_uses(
              frame.schedule.placement_pair_uses, repair_arena,
              &pair_replication_preferred_pairs));
          const loom_low_pressure_cliff_table_t pressure_cliffs =
              frame_options->pressure_cliffs;
          if (pressure_cliffs.count != 0) {
            IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
                repair_arena, pressure_cliffs.count,
                sizeof(*pair_replication_baseline_cliff_units),
                (void**)&pair_replication_baseline_cliff_units));
            loom_low_emission_frame_capture_pressure_cliff_units(
                pressure_cliffs, &frame.allocation,
                pair_replication_baseline_cliff_units);
          }
          pair_replication_baseline_packet_move_count =
              frame.allocation.packet_move_count;
          ++repair_iteration_count;
          restore_frame_before_build = true;
          continue;
        }
      }
      loom_low_emission_frame_materialize_address_state_result_t
          address_state_result = {0};
      IREE_RETURN_IF_ERROR(loom_low_emission_frame_materialize_address_state(
          spill_free_options, module, low_func_op, &frame, scratch_arena,
          &address_state_result));
      if (address_state_result.error_count != 0) {
        return iree_ok_status();
      }
      if (address_state_result.changed) {
        if (repair_iteration_count >=
            LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS) {
          return loom_low_emission_frame_fail_final(
              frame_options, &frame,
              LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT,
              repair_iteration_count,
              LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS);
        }
        ++repair_iteration_count;
        restore_frame_before_build = true;
        continue;
      }
      bool accepted = false;
      IREE_RETURN_IF_ERROR(loom_low_emission_frame_validate_final(
          frame_options, spill_free_options, &frame, scratch_arena, &accepted));
      if (!accepted) {
        return iree_ok_status();
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
          repair_iteration_count,
          LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS);
    }
    if (repair_iteration_count <
            LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS &&
        frame.allocation.spill_plan_count < last_repaired_spill_plan_count) {
      loom_low_allocation_rematerialization_result_t rematerialization_result =
          {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_spill_plan(
          module, &frame.allocation, scratch_arena, &rematerialization_result));
      if (rematerialization_result.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_emission_frame_emit_rematerialization_decision(
                frame_options, &frame.allocation,
                LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_SPILL_PLAN,
                &rematerialization_result));
        last_repaired_spill_plan_count = frame.allocation.spill_plan_count;
        ++repair_iteration_count;
        restore_frame_before_build = true;
        continue;
      }
      loom_low_allocation_live_range_split_result_t split_result = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_split_fixed_value_spill_plan(
          module, &frame.allocation, scratch_arena, &split_result));
      if (split_result.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_emission_frame_emit_live_range_split_decision(
                frame_options, &frame.allocation,
                LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN,
                &split_result));
        last_repaired_spill_plan_count = frame.allocation.spill_plan_count;
        ++repair_iteration_count;
        restore_frame_before_build = true;
        continue;
      }
    }
    if (repair_iteration_count >=
        LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT,
          repair_iteration_count,
          LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS);
    }

    loom_low_allocation_materialization_result_t result = {0};
    loom_low_allocation_materialization_options_t materialization_options =
        spill_free_options->materialization_options;
    materialization_options.allow_existing_storage_traffic = true;
    // The frame loop materializes one complete allocation snapshot at a time;
    // the next iteration accounts for any spill traffic introduced by it.
    materialization_options.max_spill_plan_count = 0;
    if (materialization_options.emitter.fn == NULL) {
      materialization_options.emitter = frame_options->emitter;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_spills(
        module, &frame.allocation, &materialization_options, scratch_arena,
        &result));
    if (result.error_count != 0) {
      return iree_ok_status();
    }
    if (result.storage_count == 0 && result.spill_count == 0 &&
        result.reload_count == 0) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS,
          repair_iteration_count,
          LOOM_LOW_EMISSION_FRAME_MAX_REPAIR_ITERATIONS);
    }
    loom_low_emission_frame_accumulate_materialization(
        &result, &materialization_summary);
    IREE_RETURN_IF_ERROR(
        loom_low_emission_frame_append_materialized_spill_records(
            result.materialized_spills, result.materialized_spill_count,
            &materialization_summary.spill_records, repair_arena));
    last_repaired_spill_plan_count = IREE_HOST_SIZE_MAX;

    iree_arena_reset(scratch_arena);
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_lower_spill_traffic(
        frame_options, spill_free_options, module, low_func_op,
        &spill_lowering_result, scratch_arena));
    if (spill_lowering_result.error_count != 0) {
      return iree_ok_status();
    }
    ++repair_iteration_count;
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
  const iree_arena_checkpoint_t frame_checkpoint =
      iree_arena_checkpoint_save(arena);
  iree_arena_allocator_t repair_arena;
  iree_arena_initialize(arena->block_pool, &repair_arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status = loom_low_emission_frame_build_spill_free_impl(
      module, low_func_op, frame_options, spill_free_options, &frame_checkpoint,
      &repair_arena, &scratch_arena, arena, out_frame);
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&repair_arena);
  return status;
}
